const resultElement = document.getElementById('result');
const cat = document.getElementById('cat');

const resultImage = document.getElementById('result01');

let dogFileNames;
let dogFilterNames;

fetch('./tfjs_model/dog/dog_files.txt')
  .then(response => response.text())
  .then(text => dogFileNames = text.split('\n'))

fetch('./tfjs_model/dog/dog_filter_names.txt')
  .then(response => response.text())
  .then(text => dogFilterNames = text.split('\n'))

cat.onload = async () => {
    resultElement.innerText = 'Loading MePet...';
    console.time('Loading of model');
    const model = await tf.loadGraphModel("./tfjs_model/dog/model.json");
    console.timeEnd('Loading of model');

    resultElement.innerText = 'Predicting ...';

    let imgs = tf.image.resizeBilinear(tf.browser.fromPixels(cat).expandDims(0), [224, 224]);
    imgs = imgs.mean(3);
    imgs = tf.stack([imgs, imgs, imgs], 3);
    imgs = tf.sub(tf.div(tf.cast(imgs, 'float32'), 127.5), 1);

    console.time('First prediction');

    let result = model.execute(imgs);
    let featDiffs = result[1].arraySync()[0];
    let result_idx = result[0].arraySync()[0];

    console.log("result0: ", featDiffs);
    console.log("result1: ", result_idx);

    console.timeEnd('First prediction');

    let i;
    let max_img = 50;
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

        resultNode.innerHTML = i + " :: " + dogFileNames[result_idx[i]] + " :: " + featDiffs[result_idx[i]] + "<br />";
        resultNode.appendChild(imgNode);

        resultImage.appendChild(resultNode);

        img_show++;
        if(img_show > max_img) {
            break;
        }
    }
    resultElement.innerText = 'Done!';
};
