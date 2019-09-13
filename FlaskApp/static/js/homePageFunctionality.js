$(function(){
	$('#btnAnalyze').click(function(){
		
		$.ajax({
			url: '/userHome/analysis',
			data: $('form').serialize(),
			type: 'ANALYZE',
			success: function(response){
				console.log(response);
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});