let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');

//let 
let save = async (tensor, name) => {
    let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100)
    await fs.promises.writeFile(name, bin);
};
/**
 * 
 * @param {tf.Tensor} img 
 */
let fdetect = (imgg, count) => {
    let img = tf.conv2d(imgg, tf.concat(new Array(imgg.shape[2]).fill(0).map(v => tf.tensor([0.5, 1, 0.5, 1, -6, 1, 0.5, 1, 0.5], [3, 3, 1, 1])), -1).reshape([3, 3, imgg.shape[2], 1]), 1, 'valid').relu()

    let w = img.shape[1]
    let h = img.shape[0]
    let data = img.dataSync();
    let ndata = new Float32Array((w - 2) * (h - 2) * 5)
    for (let x = 1; x < (w - 1); x++) {
        for (let y = 1; y < (h - 1); y++) {
            let k = data[(y - 0) * w + (x - 0)]
            let pos = (y * (w - 2)) * 5 + x * 5
            ndata[pos + 0] = Math.abs(data[(y - 1) * w + (x - 0)] - data[(y + 1) * w + (x - 0)]) * k
            ndata[pos + 1] = Math.abs(data[(y - 1) * w + (x - 1)] - data[(y + 1) * w + (x + 1)]) * k
            ndata[pos + 2] = Math.abs(data[(y - 0) * w + (x - 1)] - data[(y + 0) * w + (x + 1)]) * k
            ndata[pos + 3] = Math.abs(data[(y + 1) * w + (x - 1)] - data[(y - 1) * w + (x + 1)]) * k
            ndata[pos + 4] = (ndata[pos + 0] + ndata[pos + 1] + ndata[pos + 2] + ndata[pos + 3]) / 4

        }
    }
    img = tf.tensor(ndata, [h - 2, w - 2, 5])//.pool([2, 2], 'avg', 'valid', 1, 2)
    img = img.slice([0, 0, 4], [img.shape[0], img.shape[1], 1])


    let descs = Array.from(img.dataSync()).map((v, i) => ({ i, v }))
    descs = descs.map((v, i, arr) => (arr[i * 2] && arr[i * 2].v) > (arr[i * 2 + 1] && arr[i * 2 + 1].v) ? (arr[i * 2] && arr[i * 2]) : (arr[i * 2 + 1] && arr[i * 2 + 1])).slice(0, Math.floor(descs.length / 2))
    return [img, descs.sort((l, r) => r.v - l.v).slice(0, count)]
};

; (async () => {
    let img = tf.node.decodeImage(await fs.promises.readFile('./27.png')).toFloat().div(255)
    console.log('img: ', img);


    [img, descs] = fdetect(img)

    console.log('descs: ', descs);

    //await save(img.slice([0, 0, 3], [img.shape[0], img.shape[1], 1]).mul(255), './temp/resultf.jpg')
    img = tf.concat([img, tf.zeros(img.shape), tf.zeros(img.shape)], -1)
    let imgShape = img.shape;
    data = img.mul(2.2).dataSync()
    descs.forEach(d => {
        let x = d.i % imgShape[1]
        let y = Math.round(d.i / imgShape[1])
        data[y * imgShape[1] * imgShape[2] + x * imgShape[2] + 1] = 1
        data[y * imgShape[1] * imgShape[2] + x * imgShape[2] + 2] = 1
    })
    img = tf.tensor(data, imgShape)

    await save(img.maximum(0).minimum(1).mul(255), './temp/resultf.jpg')
    //await save(tf.concat([img, img, img], -1).maximum(0).minimum(1).mul(255), './temp/resultf.jpg')


})();