let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');

tf.enableProdMode()
let inputChannels = 24 + 21
//let random = tf.rand([375,500,inputChannels], Math.random)

let w = 480;
let h = 352;
/**
 * @type {tf.Tensor}
 */
let uvdata = new Float32Array(w * h * 2)
for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
        uvdata[y * w * 2 + x * 2 + 0] = y / h
        uvdata[y * w * 2 + x * 2 + 1] = x / w
    }
}
let fuv = tf.tensor(uvdata, [h, w, 2])
uv = fuv.slice([0, 0], [96, 96])

let fuv1 = fuv.slice([0, 0, 0], [h, w, 1])
let fuv2 = fuv.slice([0, 0, 1], [h, w, 1])
let random = tf.concat([
    fuv,
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(6200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(5200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(5200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(6200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(3200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(3200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(1600).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(1600).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(1200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(800).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(800).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(800).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(800).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(800).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(800).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(400).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(400).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(400).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(400).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(400).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(400).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(200).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(100).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(100).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(100).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(50).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(50).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(50).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(50).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(25).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(25).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(25).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(12).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(12).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(12).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(6).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(6).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(6).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(3).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(3).sin(),
    fuv1.mul(Math.random()).add(fuv2.mul(Math.random() * 2 - 1)).mul(3).sin(),
], -1)

let save = async (tensor, name) => {
    let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100)
    await fs.promises.writeFile(name, bin);
}
let xys = [[0, 0], [120, 120], [255, 380], [230, 257], [150, 320], [140, 300], [120, 370], [40, 200], [50, 350], [45, 260], [150, 50], [230, 70]];

let modelPrep = () => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 96, activation: 'linear', inputShape: [null, null, inputChannels] }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.batchNormalization({}));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 64, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.batchNormalization({}));
    //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 64, activation: 'linear', }));
    //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    //model.add(tf.layers.batchNormalization({}));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 50, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.batchNormalization({}));
    //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 30, activation: 'linear', }));
    //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 20, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.batchNormalization({}));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 12, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.batchNormalization({}));

    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 6, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.batchNormalization({}));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 3, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

    return model;
}


let modeltest = (model) => {
    /*let uv =  new Float32Array(w*h*inputChannels)
    
    for(let y=0;y<h;y++){
        for(let x=0;x<w;x++){
            uv[y*w*2+x*2+0]=y/375
            uv[y*w*2+x*2+1]=x/500
        }
    }
    uv = tf.tensor(uv, [w*h,2])
    let result =model.predict(uv).mul(255).reshape([h,w,3])*/
    let result = tf.tidy(() => model.predict(random.expandDims()).mul(255).reshape([h, w, 3]))

    return save(result, './temp/result.jpg')
}

    ; (async () => {
        await save(random.slice([0, 0, 2], [h, w, 3]).mul(255), './temp/rand.jpg')
        let img = tf.node.decodeJpeg(await fs.promises.readFile('./im.jpg')).slice([0, 0, 0], [h, w, 3]).toFloat().div(255)


        //const vgg19 = await tf.loadLayersModel(tf.io.fileSystem('./tfjs_vgg19_imagenet/model/model.json'));

        let ys = xys.map(xy => img.slice(xy, [96, 96]).expandDims());

        let back = tf.zeros(img.shape)
        ys.forEach((img, i) => { back = back.maximum(img.reshape([96, 96, 3]).pad([[xys[i][0], (352 - 96) - xys[i][0]], [xys[i][1], (480 - 96) - xys[i][1]], [0, 0]])) })
        await save(back.mul(255), './temp/back.jpg')

        let mask = tf.zeros(img.shape.slice(0, -1).concat([1]))
        ys.forEach((img, i) => { mask = mask.maximum(tf.ones([96, 96, 1]).pad([[xys[i][0], (352 - 96) - xys[i][0]], [xys[i][1], (480 - 96) - xys[i][1]], [0, 0]])) })

        ys = [].concat(ys.map(y => y.slice([0, 0, 0], [1, 32, 96]))).concat(ys.map(y => y.slice([0, 32, 0], [1, 32, 96]))).concat(ys.map(y => y.slice([0, 64, 0], [1, 32, 96])))
        //let vggr = vgg19.predict(img.expandDims())
        //await save(vggr.slice([0,0,0,6],[1,11,15,3]).reshape([11,15,3]).resizeBilinear([h,w]).mul(255), './temp/vgg.jpg')
        //console.log('vggx :', vggx);

        let model = modelPrep()
        //const xs = tf.stack(xys.map(xy=> uv.add([xy[0]/375, xy[1]/500]) )).reshape([12*100*100,2]);
        let xs = xys.map(xy => random.slice(xy, [96, 96]).expandDims())
        xs = [].concat(xs.map(x => x.slice([0, 0, 0], [1, 32, 96]))).concat(xs.map(x => x.slice([0, 32, 0], [1, 32, 96]))).concat(xs.map(x => x.slice([0, 64, 0], [1, 32, 96])))
        //let vggy = ys.map(y=>vgg19.predict(y));

        let epochs = 1000;
        let time = new Date()
        for (let e = 0; e < epochs; e++) {
            tf.tidy(() => {
                let loss = tf.scalar(0)
                for (let i = 0; i < xs.length; i++) {
                    let terr = model.optimizer.minimize(() => {
                        let error = tf.scalar(0)

                        let res = model.predict(xs[i], { batchSize: null })

                        error = error.add(tf.losses.meanSquaredError(res, ys[i]))
                        return error;

                    }, true,
                        [].concat(...model.layers.map(l => l._trainableWeights.map(v => v.val)))
                    )

                    loss = loss.add(terr.div(xs.length))
                }

                console.log(e, loss.dataSync()[0], new Date() - time)
            })
            e % 10 == 9 && await modeltest(model)
        }

        await modeltest(model)


    })();