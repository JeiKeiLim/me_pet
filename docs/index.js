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

    model = await tf.loadGraphModel(modelURL);
}
function initMessages() {
    loadingMsg.innerHTML = "인공지능이 분석중입니다...<br />잠시만 기다려주세요!";
    resultHeadMsg.innerHTML = "나와 닮은 강아지는...?";
    resultProb.innerHTML = '';
    resultProbMsg.innerHTML = '';
    resultLabel.innerHTML = '';
    resultImage.ineerHTML = '';
    resultContents.innerHTML = '';
}

async function predict() {
    // predict can take in an image, video or canvas html element
    let imagePreview = document.getElementById('imagePreview');
    let imgs = tf.image.resizeBilinear(tf.browser.fromPixels(imagePreview).expandDims(0), [224, 224]);
    imgs = imgs.mean(3);
    imgs = tf.stack([imgs, imgs, imgs], 3);
    imgs = tf.sub(tf.div(tf.cast(imgs, 'float32'), 127.5), 1);

    const prediction = await model.execute(imgs);
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
        if(Math.random() > prob) {
            break;
        }
    }

    resultImage.src = "./images/dogs/" + dogFileNames[show_idx];

    let dogName = dogFileNames[show_idx].replace(/_[0-9]+.*/, "");

    resultLabel.innerHTML = nameConverter[dogName];
    resultProb.innerHTML = (featDiffs[show_idx]*100).toFixed(2) + "%";
    resultProbMsg.innerHTML = "확률로 일치!";
    resultContents.innerHTML = contentsConverter[dogName];
}

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imagePreview').attr('src', e.target.result);
            $('#imagePreview').hide();
            $('#imagePreview').fadeIn(650);
        };
        reader.readAsDataURL(input.files[0]);
        setTimeout(() => {
                initResult().then(() => {
                    predict();
                    loadingMsg.innerHTML = "";
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

initParameters().then(() => {});

