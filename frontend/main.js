const imgInp = document.getElementById("imgInp");
const blah = document.getElementById("prImg");
const result = document.getElementById("result");

import runai from "./disease_classifier";

const diseases = [
  "abscess",
  "ards",
  "atelectasis",
  "atherosclerosis of the aorta",
  "cardiomegaly",
  "emphysema",
  "fracture",
  "hydropneumothorax",
  "hydrothorax",
  "pneumonia",
  "pneumosclerosis",
  "post-inflammatory changes",
  "post-traumatic ribs deformation",
  "sarcoidosis",
  "scoliosis",
  "tuberculosis",
  "venous congestion",
];

imgInp.onchange = async (evt) => {
  const [file] = imgInp.files;
  if (file) {
    blah.src = URL.createObjectURL(file);
    blah.style.display = "inline";
    result.innerText = "processing...";

    runai(file).then((r = Array()) => {
      const results = r;
      const mx = Math.max(...results);
      const i = results.indexOf(mx);
      console.log(diseases[i]);

      result.innerText = `${diseases[i]} and ${i}`;
    });
  }
};
