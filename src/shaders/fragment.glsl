#version 450

layout(location = 0) out vec4 f_color;

uniform vec2 iResolution;
uniform sampler2D iChannel0;
uniform vec2 direction;

vec4 blur9(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3846153846) * direction;
  vec2 off2 = vec2(3.2307692308) * direction;
  color += texture2D(image, uv) * 0.2270270270;
  color += texture2D(image, uv + (off1 / resolution)) * 0.3162162162;
  color += texture2D(image, uv - (off1 / resolution)) * 0.3162162162;
  color += texture2D(image, uv + (off2 / resolution)) * 0.0702702703;
  color += texture2D(image, uv - (off2 / resolution)) * 0.0702702703;
  return color;
}


void main() {
//    f_color = vec4(0.2, 0.1, 0.0, 1.0);
    vec2 uv = vec2(gl_FragCoord.xy / iResolution.xy);
    gl_FragColor = blur(iChannel0, uv, iResolution.xy, direction);
}

