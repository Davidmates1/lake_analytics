//VERSION=3
const PARAMS = {
    // Indices
    chlIndex: 'mci',
    tssIndex: null,
    watermaskIndices: ['ndwi', 'hol'],
    // Limits
    chlMin: -0.005,
    chlMax: 0.05,
    tssMin: 0.075,
    tssMax: 0.185,
    waterMax: 0,
    cloudMax: 0.02,
    // Graphics
    foreground: 'default',
    foregroundOpacity: 1.0,
    background: 'black',
    backgroundOpacity: 1.0
};

let B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11;

function getIndices(t) {
    return t ? {
        natural: "[1.0*B07+1.4*B09-0.1*B14,1.1*B05+1.4*B06-0.2*B14,2.6*B04-B14*0.6]",
        chl: {
            flh: "B10-1.005*(B08+(B11-B08)*((0.681-0.665)/(0.708-0.665)))",
            rlh: "B11-B10-(B18-B10*((0.70875-0.68125)*1000.0))/((0.885-0.68125)*1000.0)",
            mci: "B11-((0.75375-0.70875)/(0.75375-0.68125))*B10-(1.0-(0.75375-0.70875)/(0.75375-0.68125))*B12"
        },
        tss: {
            b07: "B07",
            b11: "B11"
        },
        watermask: {
            ndwi: "(B06-B17)/(B06+B17)"
        }
    } : {
        natural: "[2.5*B04,2.5*B03,2.5*B02]",
        chl: {
            rlh: "B05-B04-(B07-B04*((0.705-0.665)*1000.0))/((0.783-0.665)*1000.0)",
            mci: "B05-((0.74-0.705)/(0.74-0.665))*B04-(1.0-(0.74-0.705)/(0.74-0.665))*B06"
        },
        // mci2: "B05-0.469*B04-0.536*B06"
        tss: {
            b05: "B05"
        },
        watermask: {
            ndwi: "(B03-B08)/(B03+B08)"
        }
    }
}

function blend(t, n, e, r) {
    return t.map((function (t, l) {
        return t / 100 * e + n[l] / 100 * r
    }))
}

function getAlpha(t, n, e) {
    return n + (e - n) / 2 < t ? 100 : t <= n ? 0 : t >= e ? 1 : (t - n / 2) / (e - n) * 100
}

function getColors(t, n, e, r, l) {
    let a, B;
    switch (t) {
        case "chl":
            B = [
                [.0034, .0142, .163],
                [0, .416, .306],
                [.486, .98, 0],
                [.9465, .8431, .1048],
                [1, 0, 0]
            ], l && (B = B.reverse(), e *= 10, r /= 10), a = colorBlend(n, [e, e + (r - e) / 3, (e + r) / 2, r - (r - e) / 3, r], B);
            break;
        case "tss":
            B = [
                [.961, .871, .702],
                [.396, .263, .129]
            ], a = colorBlend(n, [e, r], B)
    }
    return a
}

function isPureWater(t) {
    return t ? B06 < .319 && B17 < .166 && B06 - B16 >= .027 && B20 - B21 < .021 : B03 < .319 && B8A < .166 && B03 - B07 >= .027 && B09 - B11 < .021
}

function isCloud(t, n) {
    const e = n ? (B04 - .175) / (.39 - .175) : (B02 - .175) / (.39 - .175);
    return e > 1 || e > 0 && (B04 - B06) / (B04 + B06) > t
}

function getEval(s) {
    return eval(s)
}

function isWater(t, n, e, r, l) {
    if (0 === n.length) return !0; {
        let a = !0;
        for (let B = 0; B < n.length; B++) {
            const u = n[B];
            if ("ndwi" == u && getEval(t.ndwi) < e) {
                a = !1;
                break
            }
            if ("hol" == u && !isPureWater(l)) {
                a = !1;
                break
            }
            if ("bcy" == u && isCloud(r, l)) {
                a = !1;
                break
            }
        }
        return a
    }
}

function getBackground(t, n, e) {
    let r, l = !1;
    const a = parseInt(100 * e);
    return "default" === t || "natural" === t ? (r = getEval(n), l = !0) : r = "black" === t ? [0, 0, 0] : "white" === t ? [1, 1, 1] : getStaticColor(t), l || 1 === e ? r : blend(r, getEval(n), a, 100 - a)
}

function getForeground(t, n, e, r) {
    let l;
    const a = parseInt(100 * r);
    return l = "natural" === t ? getEval(e) : getStaticColor(t), 1 === r ? l : blend(l, n, a, 100 - a)
}

function getStaticColor(t) {
    return [t[0] / 255, t[1] / 255, t[2] / 255]
}

function getValue(t) {
    let n, e, r, l, a;
    const B = t.chlIndex,
        u = t.tssIndex,
        o = t.background,
        s = t.foreground,
        c = t.foregroundOpacity,
        i = "undefined" != typeof B18,
        d = getIndices(i),
        f = getBackground(o, d.natural, t.backgroundOpacity);
    if (!isWater(d.watermask, t.watermaskIndices, t.waterMax, t.cloudMax, i)) return f;
    if ("default" !== s) return getForeground(s, f, d.natural, c);
    let g;
    if (null !== B) {
        const r = "default" === B ? i ? "flh" : "mci" : B;
        n = getEval(d.chl[r]), e = getColors("chl", n, t.chlMin, t.chlMax, i && "flh" === r)
    }
    if (null !== u) {
        const n = "default" === u ? i ? "b11" : "b05" : u;
        r = getEval(d.tss[n]), l = getColors("tss", r, t.tssMin, t.tssMax), a = getAlpha(r, t.tssMin, t.tssMax)
    }
    g = null !== B && null !== u ? blend(l, e, a, 100 - a) : null !== B && null === u ? e : null !== u && null === B ? blend(l, f, a, 100 - a) : f;
    const h = parseInt(100 * c);
    return 1 === c ? g : blend(g, f, h, 100 - h)
}

function evaluatePixel(samples) {
    B02 = samples.B02
    B03 = samples.B03
    B04 = samples.B04
    B05 = samples.B05
    B06 = samples.B06
    B07 = samples.B06
    B08 = samples.B08
    B8A = samples.B8A
    B09 = samples.B09
    B11 = samples.B11
    return [...getValue(PARAMS), samples.dataMask];
}

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "dataMask"]
        }],
        output: {
            bands: 4
        }
    }
}