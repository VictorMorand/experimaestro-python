import os
from pathlib import Path
import threading
import re
import time
from typing import Optional, Set
import enum
import signal
import asyncio
import sys

from .workspace import Workspace
from .api import Config
from .utils import logger
from .dependencies import Dependency, DependencyStatus, Resource
from .locking import Locks, LockError, Lock
from .connectors import ProcessThreadError
from .locking import Lock

NOTIFICATIONURL_VARNAME = "XPM_NOTIFICATION_URL"


class JobState(enum.Enum):
    UNSCHEDULED = 0
    WAITING = 1
    READY = 2
    RUNNING = 3
    DONE = 4
    ERROR = 5

    def notstarted(self):
        return self.value <= JobState.READY.value

    def running(self):
        return self.value == JobState.RUNNING.value

    def finished(self):
        return self.value >= JobState.DONE.value


class JobLock(Lock):
    def __init__(self, job):
        super().__init__()
        self.job = job

    def _acquire(self):
        return self.job.state == JobState.DONE

    def _release(self):
        return False


class JobDependency(Dependency):
    def __init__(self, job):
        super().__init__(job)

    def status(self) -> DependencyStatus:
        if self.origin.state == JobState.DONE:
            return DependencyStatus.OK
        elif self.origin.state == JobState.ERROR:
            return DependencyStatus.FAIL
        return DependencyStatus.WAIT

    def lock(self):
        return JobLock(self.origin)


class Job(Resource):
    """Context of a job"""

    def __init__(
        self,
        parameters: Config,
        *,
        workspace: Workspace = None,
        launcher: "experimaestro.launchers" = None
    ):
        super().__init__()
        self.workspace = workspace or Workspace.CURRENT
        assert self.workspace is not None, "No experiment has been defined"

        self.launcher = launcher or self.workspace.launcher
        assert self.launcher is not None, (
            "No default launcher in workspace %s" % workspace
        )

        self.type = parameters.__xpmtype__
        self.name = str(self.type.identifier).rsplit(".", 1)[-1]

        self.scheduler: Optional["Scheduler"] = None
        self.parameters = parameters
        self.state: JobState = JobState.UNSCHEDULED

        # Dependencies
        self.dependencies: Set[Dependency] = set()  # as target

        # Process
        self._process = None
        self.unsatisfied = 0

        # Meta-information
        self.starttime: Optional[float] = None
        self.submittime: Optional[float] = None
        self.endtime: Optional[float] = None
        self._progress = 0.0
        self.tags = parameters.tags()

    def __str__(self):
        return "Job[{}]".format(self.identifier)

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, progress: float):
        assert progress >= 0 and progress <= 1
        self._progress = progress
        for listener in self.scheduler.listeners:
            listener.job_state(self)

    @property
    def ready(self):
        return self.state == JobState.READY

    @property
    def jobpath(self):
        return self.workspace.jobspath / str(self.type.identifier) / self.identifier

    @property
    def identifier(self):
        return self.parameters.__xpm__.identifier.hex()

    def run(self, locks):
        """Actually run the code"""
        raise NotImplementedError()

    def wait(self):
        """Waiting for job to finish"""
        with self.scheduler.cv:
            logger.debug("Waiting for job %s to finish", self.jobpath)
            self.scheduler.cv.wait_for(
                lambda: self.state in [JobState.ERROR, JobState.DONE]
            )

        logger.debug("Job %s finished with state %s", self, self.state)
        return self.state

    @property
    def process(self):
        """Returns the process"""
        if self._process:
            return self._process
        raise NotImplementedError("Not implemented")

    @property
    def pidpath(self):
        """This file contains the file PID"""
        return self.jobpath / ("%s.pid" % self.name)

    @property
    def lockpath(self):
        """This file is used as a lock for running the job"""
        return self.jobpath / ("%s.lock" % self.name)

    @property
    def donepath(self):
        """When a job has been successful, this file is written"""
        return self.jobpath / ("%s.done" % self.name)

    @property
    def failedpath(self):
        """When a job has been unsuccessful, this file is written with an error code inside"""
        return self.jobpath / ("%s.failed" % self.name)

    @property
    def stdout(self) -> Path:
        return self.jobpath / ("%s.out" % self.name)

    @property
    def stderr(self) -> Path:
        return self.jobpath / ("%s.err" % self.name)

    def dependencychanged(self, dependency, oldstatus, status):
        value = lambda s: 1 if s == DependencyStatus.OK else 0
        self.unsatisfied -= value(status) - value(oldstatus)

        logger.info("Job %s: unsatisfied %d", self, self.unsatisfied)

        if status == DependencyStatus.FAIL:
            # Job completed
            if not self.state.finished():
                self.scheduler.jobfinished(self, JobState.ERROR)

        if self.unsatisfied == 0:
            logger.info("Job %s is ready to run", self)
            self.scheduler.start(self)


class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass


class JobError(Exception):
    def __init__(self, code):
        super.__init__("Job exited with code %d", self.code)


class JobThread(threading.Thread):
    """Job starting and monitoring thread"""

    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def run(self):
        """Run a job
        
        This method will lock all the dependencies before calling `self.job.run(locks)`
        where `locks` are the taken locks
        """

        state = None
        with Locks() as locks:
            try:
                with self.job.scheduler.cv:
                    logger.info(
                        "Starting job %s with %d dependencies",
                        self.job,
                        len(self.job.dependencies),
                    )

                    for dependency in self.job.dependencies:
                        try:
                            locks.append(dependency.lock().acquire())
                        except LockError:
                            logger.warning(
                                "Could not lock %s, aborting start for job %s",
                                dependency,
                                self.job,
                            )
                            # Just stop
                            dependency.check()
                            return

                    logger.info("Running job %s", self.job)
                    self.job.scheduler.jobstarted(self.job)
                    self.job.starttime = time.time()
                    process = self.job.run(locks)

                if isinstance(process, JobState):
                    state = process
                    logger.debug("Job %s ended (state %s)", self.job, state)
                else:
                    logger.debug("Waiting for job %s process to end", self.job)

                    code = process.wait()

                    if code is None:
                        # Case where we cannot retrieve the code right away
                        if self.job.donepath.is_file():
                            code = 0
                        else:
                            code = int(self.job.failedpath.read_text())

                    logger.debug("Job %s ended with code %s", self.job, code)
                    state = JobState.DONE if code == 0 else JobState.ERROR

            except ProcessThreadError:
                # Thrown by the child process so we can exit gracefully
                logger.debug("Graceful exit")
                # We set state to none so nothing is done
                state = None
                # We prevent locks to be unlocked
                locks.detach()
                return

            except JobError:
                logger.warning("Error while running job")
                state = JobState.ERROR

            except:
                logger.warning(
                    "Error while running job (in experimaestro)", exc_info=True
                )
                state = JobState.ERROR

            finally:
                # Release locks
                locks.release()
                if state in [JobState.DONE, JobState.ERROR]:
                    self.job.scheduler.jobfinished(self.job, state)


class SignalHandler:
    def __init__(self):
        self.schedulers = set()
        signal.signal(signal.SIGINT, self)

    def add(self, scheduler):
        self.schedulers.add(scheduler)

    def remove(self, scheduler):
        self.schedulers.remove(scheduler)

    def __call__(self, signum, frame):
        """SIGINT signal handler"""
        logger.warning("Signal received")
        for scheduler in self.schedulers:
            scheduler.exitmode = True
            with scheduler.cv:
                scheduler.cv.notify_all()


class EventLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = None

    def run(self):
        logger.debug("Starting event loop thread")
        self.loop = asyncio.new_event_loop()
        self.loop.run_forever()


SIGNAL_HANDLER = None


class Scheduler:
    """Represents an experiment"""

    CURRENT = None
    EVENT_LOOP = None

    def __init__(self, name):
        # Name of the experiment
        self.name = name

        # Whether jobs are submitted
        self.submitjobs = True

        # Condition variable for scheduler access
        self.cv = threading.Condition()

        # Exit mode activated
        self.exitmode = False

        # List of all jobs
        self.jobs: Dict[str, Job] = {}

        # List of jobs
        self.waitingjobs = set()

        # Listeners
        self.listeners: Set[Listener] = set()

    def __enter__(self):
        global SIGNAL_HANDLER
        self.old_experiment = Scheduler.CURRENT
        Scheduler.CURRENT = self

        # Create an event loop for checking things
        if Scheduler.EVENT_LOOP is None:
            Scheduler.EVENT_LOOP = EventLoopThread()
            Scheduler.EVENT_LOOP.start()

        if not SIGNAL_HANDLER:
            SIGNAL_HANDLER = SignalHandler()

        SIGNAL_HANDLER.add(self)

    def __exit__(self, exc_type, *args):
        # Wait until all tasks are completed (unless an exception was thrown)
        if exc_type:
            logger.exception("Not waiting since an exception was thrown")
            # logger.warning("Not waiting since an exception was thrown [%s]", exc_type)
        else:
            self.wait()

        # Set back the old scheduler, if any
        logger.info("Exiting experiment %s", self.name)
        Scheduler.CURRENT = self.old_experiment
        SIGNAL_HANDLER.remove(self)

    def wait(self):
        # Wait until all tasks are completed
        logger.info("Waiting that experiment %s finishes", self.name)
        with self.cv:
            logger.debug("Waiting for %d jobs to complete", len(self.waitingjobs))
            self.cv.wait_for(lambda: not self.waitingjobs or self.exitmode)

    def submit(self, job: Job):
        with self.cv:
            logger.info("Submitting job %s", job)
            if self.exitmode:
                logger.warning("Exit mode: not submitting")
                return

            if job.identifier in self.jobs:
                logger.warning("Job %s already submitted", job)
                return

            job.submittime = time.time()
            job.scheduler = self

            # Add dependencies, and add to blocking resources
            job.unsatisfied = len(job.dependencies)
            for dependency in job.dependencies:
                dependency.target = job
                dependency.origin.dependents.add(dependency)
                dependency.check()

            self.jobs[job.identifier] = job
            self.waitingjobs.add(job)

            for listener in self.listeners:
                listener.job_submitted(job)

            job.state = JobState.WAITING if job.dependencies else JobState.READY

            if job.ready:
                self.start(job)

    def start(self, job: Job):
        with self.cv:
            if self.exitmode:
                logger.warning("Exit mode: not starting")
                return
            thread = JobThread(job)
            thread.daemon = True
            thread.start()

    def jobstarted(self, job: Job):
        """Called just before a job is run"""
        for listener in self.listeners:
            listener.job_state(job)

    def jobfinished(self, job: Job, state: JobState):
        """Called when the job is finished (state = error or done)"""
        with self.cv:
            job.endtime = time.time()
            job.state = state
            logger.debug("Job %s has finished (%s)", job, job.state)
            self.waitingjobs.remove(job)
            for dependency in job.dependents:
                dependency.check()
            self.cv.notify_all()

        for listener in self.listeners:
            listener.job_state(job)

    def addlistener(self, listener: Listener):
        self.listeners.add(listener)

    def removelistener(self, listener: Listener):
        self.listeners.remove(listener)


class experiment:
    """Experiment environment"""

    def __init__(self, path: Path, name: str, *, port: int = None):
        from experimaestro.server import Server

        self.workspace = Workspace(Path(path))
        self.scheduler = Scheduler(name)
        self.server = Server(self.scheduler, port) if port else None
        if self.server:
            self.workspace.launcher.setNotificationURL(self.server.getNotificationURL())

    def wait(self):
        """Wait until the running processes have finished"""
        self.scheduler.wait()

    def __enter__(self):
        if self.server:
            self.server.start()

        self.workspace.__enter__()
        self.scheduler.__enter__()
        return self

    def __exit__(self, *args):
        self.scheduler.__exit__(*args)
        self.workspace.__exit__(*args)
        if self.server:
            self.server.stop()
