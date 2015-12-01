# cljex-opencv

Clojure code to accompany the presentation "Seeing with Clojure" at Clojure Exchange 2015.

## Usage

"Useful" code in cljex-opencv.core namespace.

read a file into an openCV matrix

    (def img (read-image “filename.jpg”))

write an openCV image matrix to a PNG or JPEG file

    (write-image “filename.png” img)

The following functions are in the cljex-opencv.demo namespace and make it easier to test out the supplied algorithms.

Attempt to find a vertical(ish) line in the supplied image and draws it.

    (show-line img)

Returns true if the supplied openCV image matrix appears to contain a blurred image.

    (is-image-blurry img)

Detects faces and draws bounding boxes around faces and eyes (if eyes detected).

    (show-faces-and-eyes img)

Given a low and high HSV (Hue Saturation Value) values, draws an outline around the largest blob detected (if any) in that colour range.

    (def low-value (Scalar. 84 80 80))
    (def high-value (Scalar. 104 255 255))
    (show-blob img low-value high-value)

## References

Building openCV with java support: http://docs.opencv.org/2.4/doc/tutorials/introduction/desktop_java/java_dev_intro.html

Example openCV project in clojure: http://docs.opencv.org/2.4/doc/tutorials/introduction/desktop_java/java_dev_intro.html

openCV javadocs: http://docs.opencv.org/java/3.0.0/

Line following in python: https://youtu.be/UGj3H6ETHJg

Blob tracking in python: http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

Blur detection in python: http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

## License

Copyright © 2015 David Snowdon

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
