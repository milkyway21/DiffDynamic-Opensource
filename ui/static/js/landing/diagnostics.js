const diag=document.getElementById('diag');
const DEBUG_MODE = false; // Set to true to show debug overlay
function log(s){if(DEBUG_MODE)diag.style.display='block';diag.textContent+=s+'\n';console.log('[DIAG]',s);}
log('DOM ready...');
window.onerror=function(m,s,l,c,e){log('JS ERROR: '+m+' at '+s+':'+l);return false;};
window.addEventListener('unhandledrejection',function(e){log('PROMISE ERROR: '+e.reason);});
