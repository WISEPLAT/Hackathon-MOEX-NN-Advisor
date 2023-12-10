$(document).ready(function (e) {

	setInterval(function() {
		is_index_updated();
//		console.log(2123213124);
	},2000);

	function is_index_updated(){
    	location.reload();

// локально не будет работать, поэтому через обновление страницы
//        $.getJSON("test.json", function(json){
//            console.log('34253245234523'); // this will show the info it in firebug console
//        });
    }

});