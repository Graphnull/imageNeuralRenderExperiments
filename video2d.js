let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');

/**
 * generate random
 * @param {number} seed 
 * @returns 
 */
function mulberry32(seed) {
  let a = seed;
  return function () {
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}
tf.enableProdMode()
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

/**
 * 
 * @param {tf.Tensor} fuv 
 */
let convertInputXY = (fuv) => {
  let rand = mulberry32(0)
  let fuv1 = fuv.slice([0, 0, 0, 0], [1, fuv.shape[1], fuv.shape[2], 1])
  let fuv2 = fuv.slice([0, 0, 0, 1], [1, fuv.shape[1], fuv.shape[2], 1])
  let params = [fuv1, fuv2]
  for (let i = 3; i < 6200; i *= 2) {
    for (let a = 0; a < 3; a++) {
      params.push(
        fuv1.mul(rand()).add(fuv2.mul(rand() * 2 - 1)).mul(i).sin()
      )
    }
  }

  return tf.concat(params, -1)
}

/**
 * 
 * @param {tf.Tensor} fuv 
 */
let convertInputTime = (fuv) => {
  let rand = mulberry32(0)
  let params = [fuv]
  for (let i = 3; i < 6200; i *= 2) {
    params.push(
      fuv.mul(i).sin()
    )
  }
  return tf.concat(params, -1)
}


let random = convertInputXY(fuv.expandDims())


let inputChannels = random.shape[3]
let save = async (tensor, name) => {
  let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100)
  await fs.promises.writeFile(name, bin);
}

let xys = [
 [0,0],[5,5], [45, 75], [110, 130], [185, 160], [245, 180], [255, 210], [255, 240], [245, 270], [235, 300], [235, 330], [225, 370], [205, 360], [165, 360],
  [125, 360], [105, 360], [115, 330], [125, 290], [135, 260], [115, 250], [65, 230], [35, 250], [35, 210], [35, 180],
];

let modelTimeToXYCreate = () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 32, activation: 'linear', kernelRegularizer:'l1l2', biasRegularizer:'l1l2', inputShape: [13] }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

  model.add(tf.layers.dense({ units: 8, activation: 'linear', kernelRegularizer:'l1l2', biasRegularizer:'l1l2', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

  model.add(tf.layers.dense({ units: 2, activation: 'linear', kernelRegularizer:'l1l2', biasRegularizer:'l1l2', }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

  return model;
}

let modelCreate = () => {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 64, activation: 'linear', inputShape: [null, null, inputChannels + 13] }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

  model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 32, activation: 'linear' }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 32, activation: 'linear' }));
  //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 16, activation: 'linear' }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  //model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 8, activation: 'linear' }));
  //model.add(tf.layers.leakyReLU({ alpha: 0.01 }));
  model.add(tf.layers.conv2d({ kernelSize: [1, 1], padding: 'valid', filters: 3, activation: 'linear' }));
  model.add(tf.layers.leakyReLU({ alpha: 0.01 }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

  return model;
}
let modeltest = (model, modelTimeToXY) => {
  /*let uv =  new Float32Array(w*h*2)
  
  for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
          uv[y*w*2+x*2+0]=y/375
          uv[y*w*2+x*2+1]=x/500
      }
  }
  uv = tf.tensor(uv, [1,h,w,2])
  let codedGlobalXY = convertInputXY(uv)
  let result = model.predict(codedGlobalXY, { batchSize: null })
  result = result.reshape(result.shape.slice(1)).mul(255)
*/
  
  let result = tf.tidy(() => {

    let back = new Float32Array(h*w*3)
    let origImagesLen = xys.length;
    for (let i = 0; i < origImagesLen; i++) {

      let codedTime = convertInputTime(tf.tensor([i],[1, 1, 1])).expandDims()

      let globalXY = modelTimeToXY.predict(codedTime.reshape([1, 13])).reshape([1, 1, 1, 2]);

      let codedGlobalXY = convertInputXY(uv.add(globalXY))

      let image = model.predict(tf.concat([codedGlobalXY,tf.ones([1,96,96,13]).mul(codedTime)],-1), { batchSize: null })
      let xy = globalXY.dataSync()
      let gx = Math.round(xy[1]*w)
      let gy =  Math.round(xy[0]*h)

      i ===2 && console.log(' xy:', gx, gy);
      let imageData= image.dataSync()

      
      for(let y = Math.min(Math.max(gy, 0), h-96); y<Math.min(Math.max(gy+96, 0), h);y++){
        for(let x = Math.min(Math.max(gx, 0), w-96); x<Math.min(Math.max(gx+96, 0), w);x++){

          back[(y*w)*3+x*3+0] = imageData[(y-gy)*96*3+(x-gx)*3+0]
          back[(y*w)*3+x*3+1] = imageData[(y-gy)*96*3+(x-gx)*3+1]
          back[(y*w)*3+x*3+2] = imageData[(y-gy)*96*3+(x-gx)*3+2]

        }
      }
      //back = back.maximum(image.reshape([96, 96, 3]).pad([[xys[0], (352 - 96) - xys[0]], [xys[1], (480 - 96) - xys[1]], [0, 0]]))

    }
    return tf.tensor(back,[h,w,3]).mul(255);
  })

  return save(result, './temp/result.jpg')
}

  ; (async () => {
    await save(random.slice([0, 0, 0, 2], [1, h, w, 3]).mul(255).reshape([h, w, 3]), './temp/rand.jpg')
    let img = tf.node.decodeJpeg(await fs.promises.readFile('./im.jpg')).slice([0, 0, 0], [h, w, 3]).toFloat().div(255)

    let modelTimeToXY = modelTimeToXYCreate();
    let model = modelCreate();




    let origImages = xys.map(xy => img.slice(xy, [96, 96]).expandDims());


    //let back = tf.zeros(img.shape)
    //ys.forEach((img, i) => { back = back.maximum(img.reshape([96, 96, 3]).pad([[xys[i][0], (352 - 96) - xys[i][0]], [xys[i][1], (480 - 96) - xys[i][1]], [0, 0]])) })
    //await save(back.mul(255), './temp/back.jpg')

    //let mask = tf.zeros(img.shape.slice(0, -1).concat([1]))
    //ys.forEach((img, i) => { mask = mask.maximum(tf.ones([96, 96, 1]).pad([[xys[i][0], (352 - 96) - xys[i][0]], [xys[i][1], (480 - 96) - xys[i][1]], [0, 0]])) })

    //ys = [].concat(ys.map(y => y.slice([0, 0, 0], [1, 32, 96]))).concat(ys.map(y => y.slice([0, 32, 0], [1, 32, 96]))).concat(ys.map(y => y.slice([0, 64, 0], [1, 32, 96])))
    //let xs = xys.map(xy => random.slice(xy, [96, 96]).expandDims())
    //xs = [].concat(xs.map(x => x.slice([0, 0, 0], [1, 32, 96]))).concat(xs.map(x => x.slice([0, 32, 0], [1, 32, 96]))).concat(xs.map(x => x.slice([0, 64, 0], [1, 32, 96])))


    let epochs = 1000;
    let timeLog = new Date()
    
    let codedTimes = []
    
    for (let i = 0; i < origImages.length; i++) {
      codedTimes[i] = convertInputTime(tf.tensor([i],[1, 1, 1])).expandDims()
    }
    for (let e = 0; e < epochs; e++) {
      tf.tidy(() => {
        let loss = tf.scalar(0)
        for (let i = 0; i < origImages.length; i++) {
          let terr = model.optimizer.minimize(() => {

            let globalXY = modelTimeToXY.predict(codedTimes[i].reshape([1, 13])).reshape([1, 1, 1, 2]);

            let codedGlobalXY = convertInputXY(uv.add(globalXY))

            let image = model.predict(tf.concat([codedGlobalXY, tf.ones([1,96,96,13]).mul(codedTimes[i])],-1), { batchSize: null })

            let error = tf.scalar(0)
            error = error.add(tf.losses.meanSquaredError(image, origImages[i]))

            return error;
          }, true,
            [].concat(...model.layers.map(l => l._trainableWeights.map(v => v.val)))
          )

          let terr2 = 0
          if(e>50){
          terr2 = modelTimeToXY.optimizer.minimize(() => {

            let globalXY = modelTimeToXY.predict(codedTimes[i].reshape([1, 13])).reshape([1, 1, 1, 2]);
            
            let error = tf.scalar(0)
            error = error.add(globalXY.mul(-1).relu().mean())
            error = error.add(globalXY.sub(1).relu().mean())


            let codedGlobalXY = convertInputXY(uv.add(globalXY))

            let image = model.predict(tf.concat([codedGlobalXY, tf.ones([1,96,96,13]).mul(codedTimes[i])],-1), { batchSize: null })

            //error = error.add(tf.losses.meanSquaredError(image, origImages[i]))

            let codedTime = convertInputTime(tf.tensor([Math.random()*codedTimes.length],[1, 1, 1])).expandDims()

            let image2 = model.predict(tf.concat([codedGlobalXY, tf.ones([1,96,96,13]).mul(codedTime)],-1), { batchSize: null })


            error = error.add(tf.losses.meanSquaredError(image,image2))

            return error;
          }, true,
            [].concat(...modelTimeToXY.layers.map(l => l._trainableWeights.map(v => v.val)))
          )
        }
          loss = loss.add(terr.add(terr2).div(origImages.length))
        }

        console.log(e, loss.dataSync()[0], new Date() - timeLog)
      })
     /// await modeltest(model, modelTimeToXY)

      e % 10 == 9 && await modeltest(model, modelTimeToXY)
    }

    await modeltest(model)


  })();