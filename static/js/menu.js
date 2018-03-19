$(function() {
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
	    {title: "More", children: [
		{title: "Sub 1", cmd: "sub1"},
		{title: "Sub 2", cmd: "sub1"}
	    ]}
	],
	select: function(event, ui) {
	    switch(ui.cmd) {
	    case "fps":
		showFPS=!showFPS
		break;
	    case "buffer":
		showBuffer=!showBuffer
		break;
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
	}
    });
    updateElements()
});
function updateElements() {
    $("#fpsbody").attr("style",showFPS ? "":"display:none;")
    $("#bufferbody").attr("style",showBuffer ? "":"display:none;")

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
