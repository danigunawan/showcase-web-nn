
var video = null;
var canvas = null;
var ctx = null;
var smallCanvas = null;
var scctx = null;
$(function(){
    video = document.getElementById("videoElement");
    canvas = document.getElementById("mainCanvas");
    ctx = canvas.getContext("2d");
    smallCanvas = document.getElementById("smallCanvas");
    smallCanvas.width=cameraSizeX
    smallCanvas.height=cameraSizeY
    scctx = smallCanvas.getContext("2d");

    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;

    if (navigator.getUserMedia) {
        navigator.getUserMedia({ video: true }, handleVideo, videoError);
    }
    
    setInterval(mainLoop, 100);
    /*setInterval(function() {
        scctx.drawImage(video, 0, 0, 50, 50);
        //scctx.putImageData(toSizes(scctx.getImageData(0, 0, 50, 50)), 0, 0);
        drawBerzanGubbar(toBWArray(scctx.getImageData(0, 0, 50, 50)));
    }, 100);*/
})

function mainLoop() {
    scctx.drawImage(video, 0, 0, cameraSizeX, cameraSizeY);
    imageData=scctx.getImageData(0, 0, cameraSizeX, cameraSizeY)
    var arr = [];
    for (var i = 0; i < 50; i++) {
        arr[i] = [];
    }
    var byteArray = new Uint8Array(cameraSizeX*cameraSizeY*3);
    var j=0;
    for (var i = 0; i < imageData.height * imageData.width * 4; i += 4) {
        var r = imageData.data[i]
        var g = imageData.data[i + 1]
        var b = imageData.data[i + 2]
	byteArray[j]=r
	byteArray[j+1]=g
	byteArray[j+2]=b
	j+=3
    }
    //blob=new Blob([byteArray])
    var xhr = new XMLHttpRequest;
    xhr.open("POST", "/stream/bw/send", false);
    xhr.send(byteArray);
    /*
    var fd = new FormData();
    fd.append('data', blob);
    $.ajax({
	type: 'POST',
	url: '/stream/bw/send',
	data: fd,
	processData: false,
	contentType: false
    }).done(function(data) {
	console.log(data);
    });
    */


}

function handleVideo(stream) {
    video.src = window.URL.createObjectURL(stream);
}
function videoError(e) {
    alert("error")
    console.log(e)
}
var video = document.getElementById("videoElement");
