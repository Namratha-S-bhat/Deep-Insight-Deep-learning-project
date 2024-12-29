// main.js

$(document).ready(function() {
    $('input[type="file"]').on('change', function() {
        var fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').addClass("selected").html(fileName);
    });
});