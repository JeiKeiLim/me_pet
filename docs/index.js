const resultImage = document.getElementById('result');

let dogFileNames, dogFilterNames;
let URL = "./tfjs_model/dogs/"
let model

fetch('./tfjs_model/dogs/dog_labels.txt')
  .then(response => response.text())
  .then(text => dogFileNames = text.split('\n'))

fetch('./tfjs_model/dogs/dog_filters.txt')
  .then(response => response.text())
  .then(text => dogFilterNames = text.split('\n'))

// Load the image model and setup the webcam
async function init() {
    const modelURL = URL + "model.json";

    model = await tf.loadGraphModel(modelURL);
    resultImage.innerHTML = '';
}

async function predict() {
    // predict can take in an image, video or canvas html element
    let imagePreview = document.getElementById('imagePreview');
    let imgs = tf.image.resizeBilinear(tf.browser.fromPixels(imagePreview).expandDims(0), [224, 224]);
    imgs = imgs.mean(3);
    imgs = tf.stack([imgs, imgs, imgs], 3);
    imgs = tf.sub(tf.div(tf.cast(imgs, 'float32'), 127.5), 1);

    const prediction = await model.execute(imgs);
    let featDiffs = prediction[1].arraySync()[0];
    let result_idx = prediction[0].arraySync()[0];

    let i;
    let max_img = 3;
    let img_show = 0;
    for(i=0; i<500; i++) {
        if(dogFilterNames.includes(dogFileNames[result_idx[i]])) {
            continue;
        }

        let resultNode = document.createElement("div");
        resultNode.style.float = "left";

        let imgNode = document.createElement("img");
        imgNode.src = "./images/dogs/" + dogFileNames[result_idx[i]];
        imgNode.width = 224
        imgNode.alt = dogFileNames[result_idx[i]];

        resultNode.innerHTML = dogFileNames[result_idx[i]] + "<br />";
        resultNode.innerHTML = resultNode.innerHTML + (featDiffs[result_idx[i]]*100).toFixed(2) + "%<br />";
        resultNode.style.textAlign = "center";
        resultNode.appendChild(imgNode);

        resultImage.appendChild(resultNode);

        img_show++;
        if(img_show >= max_img) {
            break;
        }
    }
}

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imagePreview').attr('src', e.target.result);
            // $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
            $('#imagePreview').hide();
            $('#imagePreview').fadeIn(650);
        };
        reader.readAsDataURL(input.files[0]);

        init().then(() => {
            predict();
        });
    }
}

$('#imageUpload').change(function () {
    readURL(this);
});

init();
