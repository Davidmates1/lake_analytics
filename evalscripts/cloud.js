
//VERSION=3
// function setup() {
//     return {
//         input: ["B02", "B03", "B04", "CLM"],
//         output: { bands: 3 }
//     }
// }

// function evaluatePixel(sample) {
//     if (sample.CLM == 1) {
//         return [0.75 + sample.B04, sample.B03, sample.B02]
//     }
//     return [sample.B04, sample.B03, sample.B02];
// }

function setup() {
    return {
        input: ["B02", "B03", "B04", "CLM", "dataMask"],
        output: { bands: 3 }
    };
}

function evaluatePixel(sample) {
    // Si el píxel está fuera de la geometría, devolver blanco
    if (sample.dataMask === 0) {
        return [1.0, 1.0, 1.0];
    }
    // Si el píxel está nublado, ajustar el color para indicar nubes
    if (sample.CLM == 1) {
        return [
            Math.min(0.75 + sample.B04, 1.0),
            Math.min(sample.B03, 1.0),
            Math.min(sample.B02, 1.0)
        ];
    }
    // Si el píxel no está nublado, devolver valores normales
    return [
        Math.min(sample.B04, 1.0),
        Math.min(sample.B03, 1.0),
        Math.min(sample.B02, 1.0)
    ];
}