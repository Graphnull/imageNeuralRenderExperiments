let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');

tf.enableProdMode()
let inputChannels = 24 + 21
//let random = tf.rand([375,500,inputChannels], Math.random)
function mulberry32(a) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}
let rand = mulberry32(0)
let w = 256;
let h = 256;
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
let dataset = tf.tidy(()=>fs.readdirSync('./nebulas/').map(f=> 
    tf.node.decodeJpeg(fs.readFileSync('./nebulas/'+f)).resizeBilinear([256,256]).toFloat().div(255)
))
let fuv = tf.tensor(uvdata, [h, w, 2])
uv = fuv.slice([0, 0], [96, 96])

let fuv1 = fuv.slice([0, 0, 0], [h, w, 1])
let fuv2 = fuv.slice([0, 0, 1], [h, w, 1])
let random = tf.concat([
    fuv.sub(0.5).abs(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(5200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(5200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(1600).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(1600).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(1200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(800).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(400).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(200).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(100).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(100).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(100).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(50).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(25).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(25).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(25).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(12).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(12).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(12).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(6).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3).sin(),
    fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(3).sin(),
], -1)

let save = async (tensor, name) => {
    let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100)
    await fs.promises.writeFile(name, bin);
}

let modelPrep = () => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 512, activation: 'linear', inputShape: [null, null, inputChannels+dataset.length] }));
    
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 64, activation: 'linear', }));
    //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 64, activation: 'linear', }));
    //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 384, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 30, activation: 'linear', }));
    //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 196, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    //model.add(tf.layers.dropout({}));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 96, activation: 'linear', }));
    //model.add(tf.layers.dropout({}));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    

    //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 6, activation: 'linear', }));
    //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
    //model.add(tf.layers.batchNormalization({}));
    model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 3, activation: 'linear', }));
    model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });
    model.summary()
    return model;
}


let modeltest = async (model) => {
    let i =0
    let ind = new Float32Array(dataset.length)
    ind[i]=1
    let onehot = tf.tensor(ind,[1, dataset.length])
    let inp = tf.concat(new Array(256*256).fill(0).map(v=> onehot)).reshape([256,256,dataset.length])

    let result = tf.tidy(() => model.predict(tf.concat([random, inp],-1).expandDims()).mul(255).reshape([h, w, 3]))
    await model.save('file://./model')
    return await save(result, './temp/result.jpg')
}

    ; (async () => {
        await save(random.slice([0, 0, 2], [h, w, 3]).mul(255), './temp/rand.jpg')
    
        
        let model = modelPrep()
        const embed = tf.sequential();

        embed.add(tf.layers.embedding({inputDim:dataset.length, outputDim:12})) ;
        embed.summary()
    
        let epochs = 1000;
        let time = new Date()
        let inps = dataset.map((v,i)=>{
            let ind = new Float32Array(dataset.length)
            ind[i]=1
            let onehot = tf.tensor(ind,[1, dataset.length])
            let inp = tf.concat(new Array(256*256).fill(0).map(v=> onehot)).reshape([256,256,dataset.length])
            return tf.concat([random, inp],-1).expandDims()
        })
        for (let e = 0; e < epochs; e++) {
            tf.tidy(() => {
                let loss = tf.scalar(0)
                for (let i = 0; i < dataset.length; i++) {
                    let err = model.optimizer.minimize(() => {
                        //let embedDescriptor = embed.predict(onehot)

                        //console.log(' random[i]:',random.shape, embedDescriptor.shape );
                        let res = model.predict(inps[i], { batchSize: null })

                        return tf.losses.meanSquaredError(res, dataset[i].expandDims())
                    }, true,
                       // [].concat(...model.layers.map(l => l._trainableWeights.map(v => v.val)))
                    )

                    loss = loss.add(err)
                }

                console.log(e, loss.dataSync()[0]/dataset.length, new Date() - time)
            })
            e % 10 == 9 && await modeltest(model)
        }

        await modeltest(model)


    })();