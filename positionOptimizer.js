let tf = require('@tensorflow/tfjs-node')
let fs = require('fs');



let w = 480;
let h = 352;


let save = async (tensor, name) => {
  let bin = await tf.node.encodeJpeg(tensor.minimum(255).maximum(0), "rgb", 100)
  await fs.promises.writeFile(name, bin);
}

let xys = [
 [45, 75], [110, 130], [185, 160], [245, 180], [255, 210], [255, 240], [245, 270], [235, 300], [235, 330], [225, 370], [205, 360], [165, 360],
  [125, 360], [105, 360], [115, 330], [125, 290], [135, 260], [115, 250], [65, 230], [35, 250], [35, 210], [35, 180],
]

let modeltest = async  (origImages, xyWeights) => {

  let back = tf.tidy(()=>{
    let th =h*1.5
    let tw =w*1.5
  let back = tf.zeros([th,tw,3])

  origImages.forEach((img, i) => { 
    let x = Math.round(xyWeights[i*2+1])+th/3
    let y = Math.round(xyWeights[i*2+0])+tw/3

    let m = tf.zeros([96, 96, 3]).pad([
      [Math.min(Math.max(0,x),th-96), Math.max(0,Math.min((th - 96) - x, th-96))], 
    [Math.min(Math.max(0,y),tw-96), Math.max(0,Math.min((tw - 96) - y, tw-96))], 
    [0, 0]],1)

    back = back.mul(m).maximum(img.reshape([96, 96, 3]).pad([
      [Math.min(Math.max(0,x),th-96), Math.max(0,Math.min((th - 96) - x, th-96))], 
    [Math.min(Math.max(0,y),tw-96), Math.max(0,Math.min((tw - 96) - y, tw-96))], 
    [0, 0]])    ) })
    return back.mul(255)
  })
  await save(back, './temp/result.jpg')
  back.dispose()
}
/**
 * 
 * @param {tf.Tensor} img1 
 * @param {tf.Tensor} img2 
 * @param {number} dx 
 * @param {number} dy 
 * @returns 
 */
let diff = (data, data2, w,h,channels, dx,dy, debug)=>{
  //console.log('w,h :', w,h);

  let error = 0;
  let xstart = Math.max(0,-dx)
  let ystart = Math.max(0,-dy)
  let xend = w+Math.min(0,-dx)
  let yend = h+Math.min(0,-dy)
  let deb = new Float32Array(w*h*channels)

  let count = (xend-xstart)*(yend-ystart)

  let find=false
  for(let x=xstart;x<xend;x++){
    for(let y=ystart;y<yend;y++){

      let pix1 = y*w*channels+x*channels+0
      let pix2 = (y+dy)*w*channels+(x+dx)*channels+0


      if(x>w){
        console.log(' x>w :');
      }

      if(typeof data2[pix2+0] !=='number'){
        continue;
      }
      for(let c=0;c<channels;c++){
        error += Math.abs(data[pix1+c]-data2[pix2+c])
       
        if(Math.abs(data[pix1+c]-data2[pix2+c]) && debug){
          console.log('Math.abs(data[pix1+c]-data2[pix2+c]) :', Math.abs(data[pix1+c]-data2[pix2+c]), data[pix1+c], data2[pix2+c]);

          //console.log(' :',y,x );
          deb[pix1+c]= data[pix1+c]*255
          find=true
          //console.log('err :',channels, x,y,c,data[pix1+c],data2[pix2+c], Math.abs(data[pix1+c]-data2[pix2+c]) );
        }
      }
      
    }
  }
  
  find && save(tf.tensor(deb, [h,w,channels]), './deb.jpg')
 

  //console.log(' errorerrorerror:',error, count );
  return [error/(count+0.001), count]
}

let r = new Float32Array(167).map(v=>Math.random())
let indr = 0;
let getRand=()=>Math.random();

  ; (async () => {


    const vgg19 = await tf.loadLayersModel(tf.io.fileSystem('./tfjs_vgg19_imagenet/model/model.json'));

    //await save(random.slice([0, 0, 0, 37], [1, h, w, 3]).mul(255).reshape([h, w, 3]), './temp/rand.jpg')

    let img = tf.node.decodeJpeg(await fs.promises.readFile('./im.jpg')).slice([0, 0, 0], [h, w, 3]).toFloat().div(255)

    let origImages = xys.map(xy => img.slice(xy, [96, 96]).expandDims());


    let back = tf.zeros(img.shape)
    origImages.forEach((img, i) => { 
      let m = tf.zeros([96, 96, 3]).pad([[xys[i][0], (352 - 96) - xys[i][0]], [xys[i][1], (480 - 96) - xys[i][1]], [0, 0]],1)
      back = back.mul(m).maximum(img.reshape([96, 96, 3]).pad([[xys[i][0], (352 - 96) - xys[i][0]], [xys[i][1], (480 - 96) - xys[i][1]], [0, 0]])) })
    await save(back.mul(255), './temp/back.jpg')


    let xyWeights = tf.variable(tf.tensor((new Float32Array(origImages.length*2)),[origImages.length,2]), true, 'xy')
    //  xyWeights.assign(tf.tensor(xys.map(v=>[v[1]+Math.random()*5,v[0]+Math.random()*5])))
      
    let epochs = 100000;
    let timeLog = new Date()
    

    let optimizer = tf.train.adam(0.5)
    await optimizer.setWeights([{name:'xy',tensor:xyWeights} ])

    vgg19.outputs = vgg19.nodesByDepth['16'][0].outputTensors[0]
    //console.log('vgg19.outputs[0] :', vgg19.layersByDepth);

    let features = origImages//.map(v=>vgg19.execute(v,vgg19.outputs.name) );
    let wf = features[0].shape[features[0].shape.length-2];
    let hf = features[0].shape[features[0].shape.length-3];
    let cf = features[0].shape[features[0].shape.length-1];
    features = features.map(v=>v.dataSync())
    //console.log('vgg19 :', features[0].shape);
    //process.exit(0)
    //vgg19
    for (let e = 0; e < epochs; e++) {
      tf.tidy(() => {
        //let loss = tf.scalar(0)
        let grad = new Float32Array(xyWeights.shape[0]*xyWeights.shape[1])
        let xyWeightsData = xyWeights.dataSync()
        
        for (let i = 0; i < features.length-1; i++) {

          let count =0;
          for (let j = i+1; j < features.length; j++) {
            let wx = Math.round(xyWeightsData[i*2+0]-xyWeightsData[j*2+0])
            //console.log('wx :', wx, xyWeightsData[i*2+0], xyWeightsData[j*2+0]);
            let wy = Math.round(xyWeightsData[i*2+1]-xyWeightsData[j*2+1])

            let rv = Math.round(getRand()*200)
            let [err, count0] = diff(features[i], features[j],wf, hf,cf, rv+wx,0+wy)
            let lcount = count0
            let dx = rv
            let dy = 0
            rv = Math.round(getRand()*200)
            let [err1, count1] = diff(features[i], features[j],wf, hf,cf, 0+wx,rv+wy)
            if(err1<err && count1){
              dx =0
              dy=rv
              err= err1
              lcount += count1
            }
            rv = Math.round(getRand()*(-200))
            let [err2, count2] = diff(features[i], features[j],wf, hf,cf, 0+wx, rv+wy)
            if(err2<err && count2){
              dx =0
              dy=rv
              err=err2
              lcount += count2
            }
            rv = Math.round(getRand()*(-200))
            let [err3, count3] = diff(features[i], features[j],wf, hf,cf, rv+wx,0+wy)
            if(err3<err && count3){
              dx = rv
              dy=0
              err=err3
              lcount += count3
            }
            let [err4, count4] = diff(features[i], features[j],wf, hf,cf, 0+wx,0+wy)
            if(err4<err && count4){
              dx =0
              dy=0
              err=err4
              lcount += count4
            }
            if(err&& false){

            console.log(' x1111:',err, err4, count4, dx,dy, i,j , count0,);
            save(tf.tensor(features[i],[96,96,3]).mul(255),'test1.jpg')
            save(tf.tensor(features[j],[96,96,3]).mul(255),'test2.jpg')

            e = 99999999
              i=999999
              j=999999
            
              break;
            }
            if(lcount===0){
              //grad[i*2+0]-=150-xyWeightsData[i*2+0]
              //grad[i*2+1]-=150-xyWeightsData[i*2+1]
            }else{
              grad[i*2+0]-=dx*Math.max(0, err4 - err )
              grad[i*2+1]-=dy*Math.max(0, err4 - err )
              grad[j*2+0]+=dx*Math.max(0, err4 - err )
              grad[j*2+1]+=dy*Math.max(0, err4 - err )
            }
            count += lcount
            if(isNaN(grad[i*2+0])){
              throw new Error('!!!')
            }
          }
          if(count===0){
            grad[i*2+0]-=-xyWeightsData[i*2+0]
            grad[i*2+1]-=-xyWeightsData[i*2+1]
          }
        }

        //console.log('grad :', grad);
        //throw new Error('!!!')
        optimizer.applyGradients([ {name:'xy',tensor:tf.tensor(grad,[features.length,2])} ])
        

        e % 10 == 9 &&console.log(e, new Date() - timeLog)
      })
     
      e % 10 == 9 && await modeltest(origImages, xyWeights.dataSync())
    
    }



  })();