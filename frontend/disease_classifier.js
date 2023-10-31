import * as onnx from "onnxjs";
import * as ndarray from "ndarray";
import * as ops from "ndarray-ops";
import * as tf from "@tensorflow/tfjs";

const image_size = 224;

function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 1), [
    1,
    1,
    height,
    width,
  ]);

  // Normalize 0-255 to (-1)-1
  ops.divseq(dataFromImage, 128.0);
  ops.subseq(dataFromImage, 1.0);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  ops.assign(
    dataProcessed.pick(0, 0, null, null),
    dataFromImage.pick(null, null, 2)
  );
  ops.assign(
    dataProcessed.pick(0, 1, null, null),
    dataFromImage.pick(null, null, 1)
  );
  ops.assign(
    dataProcessed.pick(0, 2, null, null),
    dataFromImage.pick(null, null, 0)
  );

  return dataProcessed.data;
}

function imgTransform(img) {
  img = tf.image
    .resizeBilinear(img, [224, 224])
    .div(tf.scalar(255))
    .sub(tf.scalar(0.46))
    .div(tf.scalar(0.226));
  // img = tf.cast(img, (dtype = "float32"));

  // /*mean of natural image*/
  // let meanRgb = 0.46;

  // /* standard deviation of natural image*/
  // let stdRgb = 0.226;

  // let indices = [tf.tensor1d([0], "int32")];

  // /* sperating tensor channelwise and applyin normalization to each chanel seperately */
  // let centeredRgb = {
  //   red: tf
  //     .gather(img, indices[0], 2)
  //     .sub(tf.scalar(meanRgb))
  //     .div(tf.scalar(stdRgb))
  //     .reshape([224, 224]),
  // };

  // /* combining seperate normalized channels*/
  // let processedImg = tf.stack([centeredRgb.red]).expandDims();
  return img;
}
const getImgData = async (file) => {
  var canvas = document.createElement("canvas");
  var context = canvas.getContext("2d");
  var img = new Image();
  var imgData = new ImageData(1, 1);
  canvas.width = image_size;
  canvas.height = image_size;

  return new Promise((res, rej) => {
    img.onload = async function () {
      context.drawImage(img, 0, 0);
      imgData = context.getImageData(0, 0, image_size, image_size);
      console.log(imgData);
      const processedData = processImgData(imgData);
      const result = await getResult(processedData);
      res(result);
    };

    const imgURL = window.URL.createObjectURL(file);
    img.src = imgURL;
  });
};
const processImgData = (imgData) => {
  //convert to grayscale using the PIL formula
  const processedData = [];
  for (let i = 0; i < imgData.data.length; i += 4) {
    processedData.push(
      (imgData.data[i + 0] * 299) / 1000 +
        (imgData.data[i + 1] * 587) / 1000 +
        (imgData.data[i + 2] * 114) / 1000
    );
  }

  console.log(processedData);

  //normalizing img data
  const dataFromImage = ndarray(new Float32Array(processedData), [
    image_size,
    image_size,
    1,
  ]);

  // Normalize 0-255 to (-1)-1
  ops.divseq(dataFromImage, 255);
  ops.subseq(dataFromImage, 0.46);
  ops.divseq(dataFromImage, 0.226);

  console.log(dataFromImage);

  return dataFromImage;
};
const getResult = async (processedData) => {
  const sess = new onnx.InferenceSession();
  await sess.loadModel("./onnx_model.onnx");

  const input = new onnx.Tensor(processedData.data, "float32", [
    1,
    1,
    image_size,
    image_size,
  ]);

  // var input = tf.browser.fromPixels(img);
  // input = tf.image.rgbToGrayscale(input);
  // input = imgTransform(input);

  const outputMap = await sess.run([input]);
  const outputTensor = outputMap.values().next().value;
  const result = Array.from(outputTensor.data);
  console.log(result);
  return result;
};
export default async function runai(file) {
  return await getImgData(file);
}
