# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from glumpy import app, gl, gloo, glm
from glumpy.transforms import Arcball, Position


def get_points(filename):
    """
    Load a depthmap from disk and convert it into a pointcloud using the camera's intrinsic parameters
    """
    depth = np.load(filename).squeeze()
    fx = 368.096588
    fy = 368.096588
    cx = 261.696594
    cy = 202.522202

    im_shape = depth.shape
    row = (np.arange(im_shape[0])[:,None] - cx) / fx * depth
    col = (np.arange(im_shape[1])[None,:] - cy) / fy * depth
    points = np.array((row,col,depth)).reshape(3,-1).swapaxes(0,1)
    return points

vertex = """
#version 120
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float linewidth;
uniform float antialias;
attribute vec4  fg_color;
attribute vec4  bg_color;
attribute float radius;
attribute vec3  position;
varying float v_pointsize;
varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
void main (void)
{
    v_radius = radius;
    v_fg_color = fg_color;
    v_bg_color = bg_color;
    gl_Position = projection * view * model * <transform(position)>;
    gl_PointSize = 2 * (v_radius + linewidth + 1.5*antialias);
}
"""

fragment = """
#version 120
uniform float linewidth;
uniform float antialias;
varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
float marker(vec2 P, float size)
{
   const float SQRT_2 = 1.4142135623730951;
   float x = SQRT_2/2 * (P.x - P.y);
   float y = SQRT_2/2 * (P.x + P.y);
   float r1 = max(abs(x)- size/2, abs(y)- size/10);
   float r2 = max(abs(y)- size/2, abs(x)- size/10);
   float r3 = max(abs(P.x)- size/2, abs(P.y)- size/10);
   float r4 = max(abs(P.y)- size/2, abs(P.x)- size/10);
   return min( min(r1,r2), min(r3,r4));
}
void main()
{
    float r = (v_radius + linewidth + 1.5*antialias);
    float t = linewidth/2.0 - antialias;
    float signed_distance = length(gl_PointCoord.xy - vec2(0.5,0.5)) * 2 * r - v_radius;
//    float signed_distance = marker((gl_PointCoord.xy - vec2(0.5,0.5))*r*2, 2*v_radius);
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);
    // Inside shape
    if( signed_distance < 0 ) {
        // Fully within linestroke
        if( border_distance < 0 ) {
            gl_FragColor = v_fg_color;
        } else {
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
        }
    // Outside shape
    } else {
        // Fully within linestroke
        if( border_distance < 0 ) {
            gl_FragColor = v_fg_color;
        } else if( abs(signed_distance) < (linewidth/2.0 + antialias) ) {
            gl_FragColor = vec4(v_fg_color.rgb, v_fg_color.a * alpha);
        } else {
            discard;
        }
    }
}
"""

theta, phi = 0,0
window = app.Window(width=800, height=800, color=(1,1,1,1))

# points = get_points("random_image_0.npy")
points = get_points("url_image_0.npy")
points[:,2] *= -1


points[np.isnan(points)] = 0
points[np.isinf(points)] = 0

points[:,0] -= points[:,0].mean()
points[:,1] -= points[:,1].mean()
points[:,2] -= points[:,2].mean()

# n = 1000000
n = points.shape[0]
program = gloo.Program(vertex, fragment, count=n)
view = np.eye(4, dtype=np.float32)
glm.translate(view, 0, 0, -20)


program['position'] = points# 0.35 * np.random.randn(n,3)
program['radius']   = np.ones((n,))#np.random.uniform(5,10,n)
program['fg_color'] = 0,0,0,1
colors = np.random.uniform(0.75, 1.00, (n, 4))
colors[:,3] = 1
program['bg_color'] = colors
program['linewidth'] = 1.0
program['antialias'] = 1.0
program['model'] = np.eye(4, dtype=np.float32)
program['projection'] = np.eye(4, dtype=np.float32)
program['view'] = view

# Pan zoom
transform = Arcball(Position(), aspect=1,znear=1,zfar=10)
program['transform'] = transform
window.attach(transform)

@window.event
def on_draw(dt):
    global theta, phi, translate
    window.clear()
    program.draw(gl.GL_POINTS)
    theta += .15
    phi += .15
    model = np.eye(4, dtype=np.float32)
    # glm.rotate(model, theta, 0, 0, 1)
    # glm.rotate(model, phi, 0, 1, 0)
    program['model'] = model

@window.event
def on_resize(width,height):
    program['projection'] = glm.perspective(45.0, width / float(height), 1.0, 1000.0)

gl.glEnable(gl.GL_DEPTH_TEST)
app.run()
