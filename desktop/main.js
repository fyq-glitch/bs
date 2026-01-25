const {app,BrowserWindow}=require('electron');
const path=require('path');
const {spawn}=require('child_process');
let backendProcess=null;
function createWindow(){
    const win =new BrowserWindow({
        width:1400,
        height:1200,
        webPreferences:{
            preload:path.join(__dirname,'preload.js'),
            nodeIntegration:false,
            contextIsolation:true,
        },
        title:"X-ray违禁品检测"
    });
    win.loadURL('http://localhost:8080');
}
app.whenReady().then(()=>{
    const uvicornPath = 'C:\\Users\\fyq\\anaconda3\\envs\\bs\\Scripts\\uvicorn.exe';
    backendProcess = spawn(uvicornPath, ['main:app', '--host', '127.0.0.1', '--port', '8000'], {
        cwd: path.resolve(__dirname, '../backend')
    });
    backendProcess.stdout.pipe(process.stdout);
    backendProcess.stderr.pipe(process.stderr);
    createWindow();
    app.on('activate',()=>{
        if (BrowserWindow.getAllWindows().length===0) createWindow();
    });
});
app.on('window-all-closed',()=>{
    if (backendProcess) backendProcess.kill();
    if (process.platform !== 'darwin') app.quit();
});