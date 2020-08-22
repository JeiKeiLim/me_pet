const resultImage = document.getElementById('resultImage');
const resultHeadMsg = document.getElementById('resultHeadMsg');
const resultLabel = document.getElementById('resultLabel');
const resultProb = document.getElementById('resultProb');
const resultProbMsg = document.getElementById('resultProbMsg');
const resultContents = document.getElementById('resultContents');
const loadingMsg = document.getElementById('loadingMsg');

let dogFileNames, dogFilterNames, uniqueDogNames, uniqueDogNamesKorean, dogContents;
let nameConverter = {};
let contentsConverter = {};
let URL = "./tfjs_model/dogs/";
const prob = 0.7;
let model;
let faceModel;

async function readTextToArray(path) {
    return fetch(path).then(response => response.text()).then(text => text.split('\n'));
}

async function initParameters() {
    dogFileNames = await readTextToArray('./tfjs_model/dogs/dog_labels.txt');
    dogFilterNames = await readTextToArray('./tfjs_model/dogs/dog_filters.txt');
    uniqueDogNames = await readTextToArray('./tfjs_model/dogs/dog_labels_unique.txt');
    uniqueDogNamesKorean = await readTextToArray('./tfjs_model/dogs/dog_labels_unique_korean.txt');
    dogContents = await readTextToArray("./tfjs_model/dogs/dog_labels_contents.txt");

    for(let i=0; i<uniqueDogNames.length-1; i++) {
        nameConverter[uniqueDogNames[i]] = uniqueDogNamesKorean[i];
        contentsConverter[uniqueDogNames[i]] = dogContents[i];
    }
}

// Load the image model and setup the webcam
async function initResult() {
    const modelURL = URL + "model.json";

    if(typeof(model) == 'undefined') {
        model = await tf.loadGraphModel(modelURL);
    }
    if(typeof(faceModel) == 'undefined') {
        faceModel = await blazeface.load();
    }
}

function initMessages() {
    loadingMsg.innerHTML = "인공지능이 분석중입니다...<br />잠시만 기다려주세요!";
    resultHeadMsg.innerHTML = "나와 닮은 강아지는...?";
    resultProb.innerHTML = '';
    resultProbMsg.innerHTML = '';
    resultLabel.innerHTML = '';
    resultImage.innerHTML = '';
    resultContents.innerHTML = '';

    $('#imagePreview').hide();
    $('#imagePreviewCanvas').fadeOut(500);
    $('#resultImage').fadeOut(500);
}

async function getFaceImage(imageSize=[224, 224]) {
    let img, facePreds, bestFace, bestProb;
    let imagePreview = document.getElementById('imagePreview');

    img = tf.browser.fromPixels(imagePreview)
    facePreds = await faceModel.estimateFaces(img, false);

    bestProb = 0;

    for(let i=0; i<facePreds.length; i++) {
        if(facePreds[i].probability > bestProb) {
            bestProb = facePreds[i].probability;
            bestFace = facePreds[i];
        }
    }

    if(bestProb < 0.98) {
        return tf.image.resizeBilinear(tf.expandDims(img, 0), imageSize);
    }

    let x1 = bestFace.topLeft[0] / img.shape[1];
    let y1 = bestFace.topLeft[1] / img.shape[0];
    let x2 = bestFace.bottomRight[0] / img.shape[1];
    let y2 = bestFace.bottomRight[1] / img.shape[0];

    return tf.image.cropAndResize(tf.expandDims(img, 0), [[y1, x1, y2, x2],], [0,], imageSize)
}

async function predict() {
    // predict can take in an image, video or canvas html element
    let modelImage = await getFaceImage([224, 224]);

    let faceImage = tf.reshape(modelImage, [224, 224, 3]);
    faceImage = tf.div(faceImage, 255.0);

    modelImage = modelImage.mean(3);
    modelImage = tf.stack([modelImage, modelImage, modelImage], 3);
    modelImage = tf.sub(tf.div(tf.cast(modelImage, 'float32'), 127.5), 1);

    const prediction = await model.execute(modelImage);
    let result01 = prediction[1].arraySync()[0];
    let result02 = prediction[0].arraySync()[0];

    let featDiffs, result_idx;
    if(result01[0] % 1 == 0) {
        result_idx = result01;
        featDiffs = result02;
    }
    else {
        result_idx = result02;
        featDiffs = result01;
    }

    let show_idx = result_idx[0];

    for(let i=0; i<result_idx.length; i++) {
        if(dogFilterNames.includes(dogFileNames[result_idx[i]])) {
            continue;
        }
        show_idx = result_idx[i];
        if(Math.random() < prob) {
            break;
        }
    }

    let dogName = dogFileNames[show_idx].replace(/_[0-9]+.*/, "");
    let probability = featDiffs[show_idx];
    resultImage.src = "./images/cropped_dogs/" + dogFileNames[show_idx];

    return {faceImage: faceImage, dogName: dogName, probability: probability};
}

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imagePreview').attr('src', e.target.result);
            $('#imagePreview').hide();
            $('#imagePreview').fadeIn(1000);
        };

        reader.readAsDataURL(input.files[0]);
        setTimeout(() => {
            initResult().then(() => {
                predict().then((result) => {
                    setTimeout(() => {
                        let canvas = document.getElementById('imagePreviewCanvas');
                        tf.browser.toPixels(result.faceImage, canvas).then(() => {});
                        let cavnasPreview = $('#imagePreviewCanvas');
                        cavnasPreview.css('width', '100%');
                        cavnasPreview.css('height', '100%');

                        loadingMsg.innerHTML = "";

                        $('#imagePreview').hide();
                        cavnasPreview.hide();
                        cavnasPreview.fadeIn(1000);

                        $('#resultImage').hide();
                        $('#resultImage').fadeIn(1000);

                        resultLabel.innerHTML = nameConverter[result.dogName];
                        resultProb.innerHTML = (Math.round(result.probability*10000)/100) + "%";
                        resultProbMsg.innerHTML = "확률로 일치!";
                        resultContents.innerHTML = contentsConverter[result.dogName];
                    }, 1000);
                });
            });
        },
        1000
        );
    }
}

$('#imageUpload').change(function () {
    initMessages();
    readURL(this);
});

$('#imagePreviewCanvas').hide();
$('#imageUploadButton').hide();
loadingMsg.innerHTML = "로딩 중...<br />잠시만 기다려주세요!";

initParameters().then(() => {});
initResult().then(() => {
    loadingMsg.innerHTML = "";
    $('#imageUploadButton').show();
});
