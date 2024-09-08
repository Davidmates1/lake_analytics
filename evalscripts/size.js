
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "dataMask"]
        }],
        output: {
            bands: 3
        }
    };
}

function evaluatePixel(sample) {
    if (sample.dataMask === 0) {
        return [1.0, 1.0, 1.0];
    }
    return [sample.B04, sample.B03, sample.B02];
}