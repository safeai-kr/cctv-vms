const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let video;

async function setupCamera() {
    video = document.createElement('video')
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadModels() {
    const faceModel = await blazeface.load({ inputSize: 128, doubleScoreThreshold: 0.5 });
    // const emotionModel = await tf.loadLayersModel('http://localhost:8000/emotion/model.json');
    // const agemodel = await tf.loadLayersModel('http://localhost:8000/age_models/model.json')
    const agemodel = await tf.loadLayersModel("age/model.json", { inputShape: [1, 180, 180, 1] });
    const gendermodel = await tf.loadLayersModel("gender/model.json", { inputShape: [1, 180, 180, 1] });
    return { faceModel, agemodel, gendermodel };
}

function drawLabel(ctx, x, y, label) {
    ctx.fillStyle = 'blue';
    ctx.fillRect(x, y - 20, ctx.measureText(label).width + 10, 20);
    ctx.fillStyle = 'white';
    ctx.fillText(label, x + 5, y - 5);
}

function getAgeLabel(index) {
    const ageRanges = ['1-15', '16-30', '31-45', '46-60'];
    return ageRanges[index] || 'Unknown';
}

async function detect({ faceModel, agemodel, gendermodel }) {
    const predictions = await faceModel.estimateFaces(video, false);

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
        const start = predictions[i].topLeft;
        const end = predictions[i].bottomRight;
        const size = [end[0] - start[0], end[1] - start[1]];

        const intstart = [Math.round(start[0]), Math.round(start[1])];
        // const intsize = [Math.round(size[0]), Math.round(size[1])];
        const intsize = [
            Math.min(Math.round(size[0]), canvas.width - intstart[0]),
            Math.min(Math.round(size[1]), canvas.height - intstart[1])
        ];


        ctx.beginPath();
        ctx.rect(intstart[0], intstart[1], intsize[0], intsize[1]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'blue';
        ctx.stroke();

        const faceAgeGender = tf.browser.fromPixels(video)
            .slice([intstart[1], intstart[0], 0], [intsize[1], intsize[0], 3])
            .resizeBilinear([180, 180])
            .mean(2)
            .expandDims(-1)
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims(0);

        // const faceage = tf.browser.fromPixels(video)
        //     .slice([intstart[1], intstart[0], 0], [intsize[1], intsize[0], 3])
        //     .resizeNearestNeighbor([64, 64])
        //     .toFloat()
        //     .div(tf.scalar(255.0))
        //     .expandDims(0);

        const agepred = await agemodel.predict(faceAgeGender);
        // console.log("age: ", agepred.dataSync())

        const genderpred = await gendermodel.predict(faceAgeGender);
        // console.log("gen: ", genderpred.dataSync())

        const ageArray = agepred.dataSync();
        const maxAgeIndex = ageArray.indexOf(Math.max(...ageArray));
        const predictedAge = getAgeLabel(maxAgeIndex);

        // const predictedAge = Math.round(agepred.dataSync()[0]);
        const genderLabel = genderpred.dataSync()[0] > 0.537 ? 'Female' : 'Male';

        const label = `${genderLabel}, ${predictedAge}`;

        drawLabel(ctx, intstart[0], intstart[1], label);
    }

    requestAnimationFrame(() => detect({ faceModel, agemodel, gendermodel }));
}

(async function main() {
    await setupCamera();
    video.play();

    const models = await loadModels();
    detect(models);
})();
