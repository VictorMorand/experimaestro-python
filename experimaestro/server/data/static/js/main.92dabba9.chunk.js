(this.webpackJsonpapp=this.webpackJsonpapp||[]).push([[0],{161:function(e,t,a){e.exports=a(300)},166:function(e,t,a){},172:function(e,t,a){},300:function(e,t,a){"use strict";a.r(t);var n=a(0),r=a.n(n),o=a(9),c=a.n(o),s=(a(166),a(24)),i=a(31),l=a(33),u=a(32),d=a(34),p=a(44),b=a(25),f=(a(171),a(172),a(67)),m=a(78),h=a.n(m),g=a(6),y=a(344),j=a(145),v=a.n(j),O=a(113),E=a.n(O),w=a(79),k=a(110),N=a.n(k),C=a(146),x=a.n(C),S=a(111),I=a.n(S),T=function(e){return r.a.createElement(x.a,{muiTheme:I()(N.a)},r.a.createElement(r.a.Fragment,null,e.children))},D=function(e){function t(){return Object(s.a)(this,t),Object(l.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(d.a)(t,e),Object(i.a)(t,[{key:"render",value:function(){var e=this.props,t=e.okLabel,a=void 0===t?"OK":t,n=e.cancelLabel,o=void 0===n?"Cancel":n,c=e.title,s=e.confirmation,i=e.show,l=e.proceed,u=e.dismiss,d=e.cancel,p=e.modal,b=[r.a.createElement(E.a,{label:o,secondary:!0,onClick:d}),r.a.createElement(E.a,{label:a,primary:!0,onClick:l})];return r.a.createElement(T,null,r.a.createElement(v.a,{title:c,actions:b,modal:p,open:i,onRequestClose:u},s))}}]),t}(r.a.Component),P=Object(w.confirmable)(D);function A(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function L(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?A(a,!0).forEach((function(t){Object(f.a)(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):A(a).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}var R=Object(w.createConfirmation)(P),W=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{};return R(L({confirmation:e},t))},_=a(37),J=a(150),F=a(147),M=a(70),B=a.n(M),U=(a(292),a(86)),q=B.a.mark(X),V=B.a.mark(K);function X(e){return B.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:if(!e.payload){t.next=3;break}return t.next=3,re.send({type:"refresh"});case 3:case"end":return t.stop()}}),q)}function H(e){Object(b.b)("Server error: "+e.payload,{type:"error"})}function K(){return B.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,Object(U.a)([Object(U.b)("CONNECTED",X),Object(U.b)("SERVER_ERROR",H)]);case 2:case"end":return e.stop()}}),V)}var Q=a(148),$=a(149),z=a.n($),G={connected:!1,jobs:{byId:{},ids:[]},experiments:{},experiment:null},Y=function(e){switch(e){case"running":return 5;case"error":case"waiting":return 2;case"ready":return 1;case"done":default:return 0}},Z=function(e){return function(t,a){var n=e[t],r=e[a],o=Y(r.status)-Y(n.status);return 0!==o?o:t.localeCompare(a)}},ee=Object(Q.a)((function(e,t){switch(t.type){case"CONNECTED":e.connected=t.payload;break;case"CLEAN_INFORMATION":e.jobs={byId:{},ids:[]},e.experiments={};break;case"EXPERIMENT_SET_MAIN":e.experiment=t.payload;break;case"EXPERIMENT_ADD":e.experiments[t.payload.name]=t.payload;break;case"JOB_ADD":void 0===e.jobs.byId[t.payload.jobId]&&e.jobs.ids.push(t.payload.jobId),e.jobs.byId[t.payload.jobId]=t.payload,e.jobs.ids.sort(Z(e.jobs.byId));break;case"JOB_UPDATE":void 0===e.jobs.byId[t.payload.jobId]||z.a.merge(e.jobs.byId[t.payload.jobId],t.payload),e.jobs.ids.sort(Z(e.jobs.byId))}}),G),te=Object(J.a)(),ae=Object(_.createStore)(ee,G,Object(F.composeWithDevTools)(Object(_.applyMiddleware)(te)));te.run(K);var ne=ae,re=new function e(){var t=this;Object(s.a)(this,e),this.open=function(){ne.dispatch({type:"CONNECTED",payload:!0})},this.close=function(){ne.dispatch({type:"CONNECTED",payload:!1}),Object(b.b)("Websocket connexion closed",{type:"info"})},this.message=function(e){ne.dispatch(JSON.parse(e.data))},this.send=function(e,a){return t.ws.readyState===WebSocket.OPEN?t.ws.send(JSON.stringify(e)):(console.log("Connection not ready"),a&&Object(b.b)("No websocket connection: could not "+a,{type:"error"}),!1)},this.query=function(e){arguments.length>1&&void 0!==arguments[1]&&arguments[1];return t.ws.send(JSON.stringify(e))},console.log("Connecting to websocket");var a=window.location,n="ws://"+a.hostname+(a.port?":"+a.port:"")+"/ws";this.ws=new WebSocket(n),this.ws.addEventListener("open",this.open),this.ws.addEventListener("close",this.close),this.ws.addEventListener("message",this.message)},oe=a(347),ce=a(338),se=a(335),ie=a(342),le=a(343),ue=a(339),de=a(340),pe=a(341),be=a(87),fe=a(334),me=a(114);function he(e){return r.a.createElement(fe.a,Object.assign({direction:"up"},e))}var ge=function(e,t){return r.a.createElement(se.a,null,r.a.createElement(ce.a,{primary:e,secondary:r.a.createElement(h.a,{className:"clipboard",style:{textAlign:"left"},"data-clipboard-text":t,onSuccess:function(){return b.b.success("Value copied")}},t)}))},ye=function(e){function t(){var e,a;Object(s.a)(this,t);for(var n=arguments.length,r=new Array(n),o=0;o<n;o++)r[o]=arguments[o];return(a=Object(l.a)(this,(e=Object(u.a)(t)).call.apply(e,[this].concat(r)))).state={open:!1},a}return Object(d.a)(t,e),Object(i.a)(t,[{key:"render",value:function(){var e=this.props,t=e.classes,a=e.job;return r.a.createElement("div",null,r.a.createElement(oe.a,{fullScreen:!0,open:!0,onClose:this.props.handleClose,TransitionComponent:he},r.a.createElement(ue.a,{className:t.appBar},r.a.createElement(de.a,null,r.a.createElement(pe.a,{color:"inherit",onClick:this.props.handleClose,"aria-label":"Close"},r.a.createElement("i",{className:"far fa-window-close"})),r.a.createElement(be.a,{variant:"h6",color:"inherit",className:t.flex},"Task ",a.taskId))),r.a.createElement(ie.a,null,ge("Status",a.status),ge("Path",a.locator),r.a.createElement(le.a,null),ge("Submitted",me.DateTime.fromMillis(1e3*a.submitted).toLocaleString(me.DateTime.DATETIME_FULL_WITH_SECONDS)),ge("Start",a.start),ge("End",a.end),ge("Tags",a.tags))))}}]),t}(r.a.Component),je=Object(g.a)({appBar:{position:"relative"},flex:{flex:1}})(ye);function ve(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function Oe(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?ve(a,!0).forEach((function(t){Object(f.a)(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):ve(a).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}var Ee=function(e){function t(){var e,a;Object(s.a)(this,t);for(var n=arguments.length,r=new Array(n),o=0;o<n;o++)r[o]=arguments[o];return(a=Object(l.a)(this,(e=Object(u.a)(t)).call.apply(e,[this].concat(r)))).state={searchtask:"",searchtagstring:"",searchtags:[],jobId:null},a.kill=function(e){W("Are you sure to kill this job?").then((function(){re.send({type:"kill",payload:e},"cannot kill job "+e)}),(function(){b.b.info("Action cancelled")}))},a.details=function(e){e&&re.send({type:"details",payload:e},"cannot get details for job "+e),a.setState(Oe({},a.state,{jobId:e}))},a.handleChange=function(e){return function(t){a.setState(Oe({},a.state,Object(f.a)({},e,t.target.value)))}},a.handleTagChange=function(e){for(var t=e.target.value,n=/(\S+):(?:([^"]\S*)|"([^"]+)")\s*/g,r=[],o=[];null!==(r=n.exec(t));)o.push({tag:r[1],value:r[2]});a.setState(Oe({},a.state,{searchtagstring:t,searchtags:o}))},a}return Object(d.a)(t,e),Object(i.a)(t,[{key:"render",value:function(){var e=this,t=this.props.jobs,a=this.state,n=a.searchtask,o=a.searchtags,c=a.jobId;if(c){var s=t.byId[c];return r.a.createElement(je,{job:s,handleClose:function(){return e.setState(Oe({},e.state,{jobId:null}))}})}return r.a.createElement("div",{id:"resources"},r.a.createElement(T,null,r.a.createElement("div",{className:"search"},r.a.createElement(y.a,{label:"Task",className:"textField",value:this.state.searchtask,onChange:this.handleChange("searchtask"),margin:"normal",helperText:"Task contains..."}),r.a.createElement(y.a,{label:"Tags",className:"textField",value:this.state.searchtagstring,onChange:this.handleTagChange,margin:"normal",helperText:"Search tags (format tag:value)"})),t.ids.map((function(a){var c=t.byId[a];if(""!==n&&-1===c.taskId.search(n))return null;var s=!0,i=!1,l=void 0;try{e:for(var u,d=o[Symbol.iterator]();!(s=(u=d.next()).done);s=!0){var p=u.value,f=p.tag,m=p.value,g=!0,y=!1,j=void 0;try{for(var v,O=c.tags[Symbol.iterator]();!(g=(v=O.next()).done);g=!0){var E=v.value;if(-1!==E[0].search(f)&&-1!==E[1].toString().search(m))continue e}}catch(w){y=!0,j=w}finally{try{g||null==O.return||O.return()}finally{if(y)throw j}}return null}}catch(w){i=!0,l=w}finally{try{s||null==d.return||d.return()}finally{if(i)throw l}}return r.a.createElement("div",{className:"resource",key:a},"running"===c.status?r.a.createElement(r.a.Fragment,null,r.a.createElement("span",{className:"status progressbar-container",title:"".concat(100*c.progress,"%")},r.a.createElement("span",{style:{right:"".concat(100*(1-c.progress),"%")},className:"progressbar"}),r.a.createElement("div",{className:"status-running"},c.status)),r.a.createElement("i",{className:"fa fa-skull-crossbones action",onClick:function(){return e.kill(a)}})):r.a.createElement("span",{className:"status status-".concat(c.status)},c.status),r.a.createElement("i",{className:"fas fa-eye action",title:"Details",onClick:function(){return e.details(a)}}),r.a.createElement("span",{className:"job-id"},r.a.createElement(h.a,{className:"clipboard","data-clipboard-text":"".concat(c.taskId,"/").concat(c.jobId),onSuccess:function(){return b.b.success("Job path copied")}},c.taskId)),c.tags.map((function(e){return r.a.createElement("span",{key:e[0],className:"tag"},r.a.createElement("span",{className:"name"},e[0]),r.a.createElement("span",{className:"value"},e[1]))})))}))))}}]),t}(n.Component),we=Object(p.b)((function(e){return{jobs:e.jobs}}))(Object(g.a)((function(e){return{container:{display:"flex",flexWrap:"wrap"},textField:{marginLeft:e.spacing.unit,marginRight:e.spacing.unit,width:200,padding:10},dense:{marginTop:19},menu:{width:200}}}))(Ee)),ke=function(e){function t(){return Object(s.a)(this,t),Object(l.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(d.a)(t,e),Object(i.a)(t,[{key:"render",value:function(){return r.a.createElement("div",null)}}]),t}(n.Component),Ne=Object(p.b)((function(e){return{experiment:e.experiment}}),{})(ke),Ce=function(e){function t(){return Object(s.a)(this,t),Object(l.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(d.a)(t,e),Object(i.a)(t,[{key:"render",value:function(){var e=this.props,t=e.connected,a=e.experiment;return r.a.createElement("div",null,r.a.createElement("header",{className:"App-header"},r.a.createElement("h1",{className:"App-title"},"Experimaestro ",a?" \u2013 "+a:"","  ",r.a.createElement("i",{className:"fab fa-staylinked ws-status ".concat(t?"ws-link":"ws-no-link")})," ")),r.a.createElement(b.a,{toastClassName:"dark-toast"}),r.a.createElement(Ne,null),r.a.createElement(we,null))}}]),t}(n.Component),xe=Object(p.b)((function(e){return{connected:e.connected,experiment:e.experiment}}))(Ce),Se=Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));function Ie(e){navigator.serviceWorker.register(e).then((function(e){e.onupdatefound=function(){var t=e.installing;t.onstatechange=function(){"installed"===t.state&&(navigator.serviceWorker.controller?console.log("New content is available; please refresh."):console.log("Content is cached for offline use."))}}})).catch((function(e){console.error("Error during service worker registration:",e)}))}var Te=document.getElementById("root");if(Te){c.a.render(r.a.createElement(p.a,{store:ne},r.a.createElement(xe,null)),Te),function(){if("serviceWorker"in navigator){if(new URL("",window.location).origin!==window.location.origin)return;window.addEventListener("load",(function(){var e="".concat("","/service-worker.js");Se?(!function(e){fetch(e).then((function(t){404===t.status||-1===t.headers.get("content-type").indexOf("javascript")?navigator.serviceWorker.ready.then((function(e){e.unregister().then((function(){window.location.reload()}))})):Ie(e)})).catch((function(){console.log("No internet connection found. App is running in offline mode.")}))}(e),navigator.serviceWorker.ready.then((function(){console.log("This web app is being served cache-first by a service worker. To learn more, visit https://goo.gl/SC7cgQ")}))):Ie(e)}))}}()}}},[[161,1,2]]]);
//# sourceMappingURL=main.92dabba9.chunk.js.map