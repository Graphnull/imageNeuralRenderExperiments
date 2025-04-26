



let points = [[0, 0], [1, 1], [2, 0], [3, 0.5]]

let camerasxy = [[0, -1], [1.5, -1], [3, -1]]


let getProjection = () => { }


let globalCoords2local = () => { }

let localCoords2global = () => { }


let linesForCamera = (camera, points) => { }

let getCameraPositions = (camerasxy, lines) => {

}

let main = () => {


    let lines = camerasxy.map(camera => linesForCamera(camera, points))


    let positions = getCameraPositions(camerasxy, lines)
    console.log('positions :', positions);

}

