$(function() {
    menuModels=[]
    i=0;
    while (i<models.length) {
	menuModels.push({"title":models[i],"action":function(){window.location.href = "/"+this}.bind(models[i])})
	i++
    }
    $(document).contextmenu({
	delegate: "#container",
	autoFocus: true,
	preventContextMenuForPopup: true,
	preventSelect: true,
	taphold: true,
	menu: [
	    {title: "Copy", cmd: "copy", uiIcon: "ui-icon-copy"},
	    {cmd: "fps"},
	    {cmd: "buffer"},
	    {title: "----"},
	    {title: "Facenet", children:[
		{cmd:"only_anchors"},
		{cmd:"use_nms"}
	    ]},
	    {title: "Models", children: menuModels}
	],
	select: function(event, ui) {
	    switch(ui.cmd) {
	    case "fps":
		showFPS=!showFPS
		break;
	    case "buffer":
		showBuffer=!showBuffer
		break;
	    case "only_anchors":
		//config["only_anchors"]=!config["only_anchors"]
		setConfig("only_anchors",!config["only_anchors"]? 1:0)
		break
	    case "use_nms":
		//config["use_nms"]=!config["use_nms"]
		setConfig("use_nms",!config["use_nms"]? 1:0)
		break
	    default:
		break;
	    }
	    updateElements()
	    //alert("select " + ui.cmd + " on " + ui.target.text());
	},
	beforeOpen: function(event, ui) {
	    var $menu = ui.menu,
		$target = ui.target,
		extraData = ui.extraData; // passed when menu was opened by call to open()
	    
	    // console.log("beforeOpen", event, ui, event.originalEvent.type);
	    
	    $(document)
	    //        .contextmenu("replaceMenu", [{title: "aaa"}, {title: "bbb"}])
	    //        .contextmenu("replaceMenu", "#options2")
	    //        .contextmenu("setEntry", "cut", {title: "Cuty", uiIcon: "ui-icon-heart", disabled: true})
		.contextmenu("setEntry", "fps", {uiIcon: checkBox(showFPS), title:"Show fps"})
		.contextmenu("setEntry", "buffer", {uiIcon: checkBox(showBuffer), title:"Show buffer"})
		.contextmenu("setEntry", "only_anchors", {uiIcon: checkBox(config["only_anchors"]), title:"Only anchors"})
		.contextmenu("setEntry", "use_nms", {uiIcon: checkBox(config["use_nms"]), title:"Use nms"})
	}
    });
    updateElements()
});
function updateElements() {
    $("#fpsbody").attr("style",showFPS ? "":"display:none;")
    $("#bufferbody").attr("style",showBuffer ? "":"display:none;")

}
function setConfig(key, value) {
    var xhr = new XMLHttpRequest;
    xhr.open("POST", "/stream/"+cid+"/setconf/"+key, false);
    //xhr.send(imageData.data);
    xhr.onload = function(e) {
	if (xhr.responseText == "1") {
	    config[key]=value
	    console.log("config changed")
	}
    }
    xhr.send(value);
}
function checkBox(f) {
    if (f) {
	return "ui-icon ui-icon-check"
    } else {
	return "ui-icon ui-icon-close"
    }
}

showFPS=false
showBuffer=false

// "ui-icon ui-icon-heart"
