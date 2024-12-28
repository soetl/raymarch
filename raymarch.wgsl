const AA: u32 = 1;
const TAU: f32 = 6.28318;

fn rot(a: f32) -> mat2x2<f32> {
    let s = sin(a);
    let c = cos(a);
    return mat2x2(c, -s, s, c);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(.5 + .5 * (b - a) / k, .0, 1.);
    return mix(b, a, h) - k * h * (1. - h);
}

fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sd_torus(p: vec3<f32>, r: vec2<f32>) -> f32 {
    let x = length(p.xz) - r.x;
    return length(vec2(x, p.y)) - r.y;
}

fn sd_box(p: vec3<f32>, s: vec3<f32>) -> f32 {
    var p0 = abs(p) - s;
    return length(max(p0, vec3(0.))) + min(max(p0.x, max(p0.y, p0.z)), .0);
}

fn min_x(d1: vec2<f32>, d2: vec2<f32>) -> vec2<f32> {
    if d1.x < d2.x {
        return d1;
    } else {
        return d2;
    };
}

fn map(pos: vec3<f32>) -> vec2<f32> {
    let t = time.elapsed;
    var res = vec2(pos.y, 0.);

    // substract
    res = min_x(res, vec2(max(
        sd_sphere(pos - vec3(.0, .3, .0), .3),
        -sd_box(pos - vec3(-.3 + sin(t) * .15, .3, 0.), vec3(.3, .15, .5))
    ), 24.));

    // rotate
    var bp = pos - vec3(-.25, .25, 1.25);
    bp = vec3(bp.xz * rot(t), bp.y).xzy;
    res = min_x(res, vec2(sd_box(bp, vec3(.25)), 17.5));

    // intersect
    res = min_x(res, vec2(max(
        sd_sphere(pos - vec3(-1.25, .3, .0), .3),
        sd_box(pos - vec3(sin(t) * .15 - 1.55, .3, 0.), vec3(.3, .15, .5))
    ), 25.));

    // scale
    let y = -fract(t) * (fract(t) - 1.);
    var tp = pos - vec3(-.6, .25 + .8 * y, -1.25);
    let squash = 1. + smoothstep(.15, .0, y) * .5;
    tp = vec3(tp.xz, tp.y * squash);
    res = min_x(res, vec2(sd_torus(tp, vec2(.3, .1)) / squash, 26.));

    // blend
    res = min_x(res, vec2(smin(
        sd_box(pos - vec3(1., .25, -1.), vec3(.25)),
        sd_sphere(pos - vec3(1. + sin(t) * .35, .25, -1. + cos(t) * .35), .25),
        .2
    ), 16.9));

    // morph
    res = min_x(res, vec2(mix(
        sd_torus(pos - vec3(1.25, .3, .6), vec2(.3, .06)),
        sd_box(pos - vec3(1.25, .3, .6), vec3(.3)),
        sin(t) * .5 + .5
    ), 55.));

    return res;
}

fn i_box(ro: vec3<f32>, rd: vec3<f32>, rad: vec3<f32>) -> vec2<f32> {
    let m = 1. / rd;
    let n = m * ro;
    let k = abs(m) * rad;
    let t1 = -n - k;
    let t2 = -n + k;
    return vec2(max(max(t1.x, t1.y), t1.z),
                min(min(t2.x, t2.y), t2.z));
}

fn raycast(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    var res = vec2(-1., -1.);

    var t_min = 1.;
    var t_max = 20.;

    // raytrace floor plane
    let tp1 = (0. - ro.y) / rd.y;
    if tp1 > 0. {
        t_max = min(t_max, tp1);
        res = vec2(tp1, 1.);
    }

    // raymarch primitives
    let tb = i_box(ro - vec3(0., .4, -.5), rd, vec3(2.5, .41, 2.5));
    if(tb.x < tb.y && tb.y > 0. && tb.x < t_max) {
        t_min = max(tb.x, t_min);
        t_max = min(tb.y, t_max);

        var t = t_min;
        for (var i = 0; i < 70 && t < t_max; i++) {
            let h = map(ro + rd * t);
            if abs(h.x) < 0.0001 * t {
                res = vec2(t, h.y);
                break;
            }
            t += h.x;
        }
    }

    return res;
}

fn calc_softshadow(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32, w: f32) -> f32 {
    var res = 1.;
    var t = t_min;

    for (var i = 0; i < 256 && t < t_max; i++) {
        let h = map(ro + rd * t).x;
        res = smin(res, h / (w * t), .9);
        t += clamp(h, .005, .5);
        if res < -1. || t > t_max { break; };
    }

    res = max(res, -1.);
    return .25 * (1. + res) * (1. + res) * (2. - res);
}

fn calc_normal(pos: vec3<f32>) -> vec3<f32> {
    let zero = min(time.frame, 0u);
    var n = vec3(0.);

    for (var i = zero; i < 4; i++) {
        let e = 0.5773 * (2. * vec3<f32>(f32(((i + 3) >> 1) & 1), f32((i >> 1) & 1), f32(i & 1)) - 1.);
        n += e * map(pos + 0.0005 * e).x;
    }

    return normalize(n);
}

fn calc_ambient_occlusion(pos: vec3<f32>, nor: vec3<f32>) -> f32 {
    let zero = min(time.frame, 0u);
    var occ = .0;
    var sca = 1.;

    for (var i = zero; i < 5u; i++) {
        let h = .01 * .12 * f32(i) / 4.;
        let d = map(pos + h * nor).x;
        occ += (h - d) * sca;
        sca *= 0.95;
        if occ > 0.35 { break; };
    }

    return clamp(1. - 3. * occ, 0., 1.) * (.5 + .5 * nor.y);
}

fn checker_gradbox(p: vec2<f32>, _dpdx: vec2<f32>, _dpdy: vec2<f32>) -> f32 {
    // filter kernel
    let w = abs(_dpdx) + abs(_dpdy) + 0.001;
    // analytical integral (box filter)
    let i = 2. * (abs(fract((p - .5 * w) * .5) - .5) - abs(fract((p + .5 * w) * .5) - .5)) / w;
    // xor pattern
    return .5 - .5 * i.x * i.y;
}

fn render(ro: vec3<f32>, rd: vec3<f32>, rdx: vec3<f32>, rdy: vec3<f32>) -> vec3<f32> {
    // background
    var col = vec3(.7, .7, .9) - vec3(max(rd.y, .0) * .3);

    // raycast scene
    let res = raycast(ro, rd);
    let t = res.x;
    let m = res.y;
    if m > -.5 {
        let pos = ro + t * rd;
        var nor = vec3(0.);
        if m < 1.5 {
            nor = vec3(0., 1., 0.);
        } else {
            nor = calc_normal(pos);
        };
        let refl = reflect(rd, nor);

        // material
        col = .2 + .2 * sin(m * 2. + vec3(0., 1., 2.));
        var ks = 1.;

        if m < 1.5 {
            // project pixel footprint into the plane
            let _dpdx = ro.y * (rd / rd.y - rdx / rdx.y);
            let _dpdy = ro.y * (rd / rd.y - rdy / rdy.y);

            let f = checker_gradbox(3. * pos.xz, 3. * _dpdx.xz, 3. * _dpdy.xz);
            col = .15 + f * vec3(.05);
            ks = .4;
        }

        let occ = calc_ambient_occlusion(pos, nor);
        
        var lin = vec3(0.);

        { // sun
            let lig = normalize(vec3(-.5, .4, -.6));
            let hal = normalize(lig - rd);
            var dif = clamp(dot(nor, lig), 0., 1.);
            dif *= calc_softshadow(pos, lig, .02, 2.5, .3);
            var spe = pow(clamp(dot(nor, hal), 0., 1.), 16.);
            spe *= dif;
            spe *= 0.04 + 0.96 * pow(clamp(1. - dot(hal, lig), 0., 1.), 5.);
            lin += col * 2.2 * dif * vec3(1.3, 1., .7);
            lin += 5. * spe * vec3(1.3, 1., 0.7) * ks;
        }
        { // sky
            var dif = sqrt(clamp(.5 + .5 * nor.y, 0., 1.));
            dif *= occ;
            var spe = smoothstep(-.2, .2, refl.y);
            spe *= dif;
            spe *= .04 + .96 * pow(clamp(1. + dot(nor, rd), 0., 1.), 5.);
            spe *= calc_softshadow(pos, refl, .02, 2.5, .05);
            lin += col * .6 * dif * vec3(.4, .6, 1.15);
            lin += 2. * spe * vec3(.4, .6, 1.3) * ks;
        }
        { // back
            var dif = clamp(dot(nor, normalize(vec3(.5, 0., .6))), 0., 1.) * clamp(1. - pos.y, 0., 1.);
            dif *= occ;
            lin += col * .55 * dif * vec3(.25, .25, .25);
        }
        { /// sss
            var dif = pow(clamp(1. + dot(nor, rd), 0., 1.), 2.);
            dif *= occ;
            lin += col * .25 * dif * vec3(1., 1., 1.);
        }

        col = lin;
        col = mix(col, vec3(.7, .7, .9), 1. - exp(-.0001 * pow(t, 3.)));
    }

    return vec3(clamp(col, vec3(0.), vec3(1.)));
}

fn set_camera(ro: vec3<f32>, ta: vec3<f32>, cr: f32) -> mat3x3<f32> {
    let cw = normalize(ta - ro);
    let cp = vec3(sin(cr), cos(cr), 0.);
    let cu = normalize(cross(cw, cp));
    let cv = cross(cu, cw);

    return mat3x3(cu, cv, cw);
}

fn color(ca: mat3x3<f32>, ro: vec3<f32>, uv: vec2<f32>, uv_x: vec2<f32>, uv_y: vec2<f32>) -> vec3<f32> {
    // focal length
    let fl = 2.5;

    // ray direction
    let rd = ca * normalize(vec3(uv, fl));

    // ray differentials
    let rdx = ca * normalize(vec3(uv_x, fl));
    let rdy = ca * normalize(vec3(uv_y, fl));

    // render
    var col = render(ro, rd, rdx, rdy);

    // gamma
    col = pow(col, vec3(1.4545));

    return col;
}

@compute @workgroup_size(16, 16)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(screen);
    let position = vec2(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    let mo = vec2<f32>(mouse.pos.xy) / vec2<f32>(screen_size.xy);
    let tm = 32. + time.elapsed * 1.5;
    let zero = min(time.frame, 0u);

    let ta = vec3(0., 0., 0.);
    var ro = ta + vec3(2., 2.2, 2.);
    ro = vec3(ro.yz * rot(-mo.y + .4), ro.x).zxy;
    ro = vec3(ro.xz * rot(time.elapsed * .2 + mo.x * TAU), ro.y).xzy;
    let ca = set_camera(ro, ta, 0.);

    let uv_x = vec2(((position + vec2(1., 0.)) * 2. - vec2<f32>(screen_size.xy)) / f32(screen_size.y));
    let uv_y = vec2(((position + vec2(0., 1.)) * 2. - vec2<f32>(screen_size.xy)) / f32(screen_size.y));

    var col = vec3(0.);
    if AA > 1 {
        for (var m = zero; m < AA; m++) {
            for (var n = zero; n < AA; n++) {
                let o = vec2(f32(m), f32(n) / f32(AA)) - vec2(.5);
                let uv = vec2(((position + o) * 2. - vec2<f32>(screen_size.xy)) / f32(screen_size.y));
                col += color(ca, ro, uv, uv_x, uv_y);
            }
        }
        col /= f32(AA * AA);
    } else {
        let uv = vec2((position * 2. - vec2<f32>(screen_size.xy)) / f32(screen_size.y));
        col = color(ca, ro, uv, uv_x, uv_y);
    }

    textureStore(screen, id.xy, vec4(col, 1.));
}
