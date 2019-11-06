$(document).ready(function(){

  $("#templateform").submit(

  function(evt){
        evt.preventDefault();
        var formData = new FormData($(this)[0]);
        $("#fileuploadStatus").html("File upload in progress ..");
         $.ajax({
             url: '/tumor_detection',
             type: 'POST',
             data: formData,
             async: false,
             cache: false,
             contentType: false,
             enctype: 'multipart/form-data',
             processData: false,
             success: function (response) {
              console.log(response);
               // var temp_data = JSON.parse(response)
               $("#fileuploadStatus").html("File upload Status: "+response.status);
               var temp_var = '';
               // for (i=0;i<response.data.length;i++){
               //   temp_var += '<p>'+response.data[i]+'</p>';
               // }
               $.each(response.data,function(index,value){
                 console.log(index,  value);
                 temp_var += '<p>'+value+'</p>';

               });
               console.log(temp_var);
               $("#apioutput").html("Image size: "+temp_var);
               $("#model_img_original").attr("src", response.file_url_original);
               $("#model_img_output").attr("src", response.file_url_output);
             }
         });
   });
});
