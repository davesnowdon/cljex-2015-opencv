(ns cljex-opencv.core
    (:import [org.opencv.core Core Point Rect Mat MatOfRect MatOfDouble MatOfPoint MatOfPoint2f CvType Size Scalar]
           org.opencv.imgcodecs.Imgcodecs
           org.opencv.imgproc.Imgproc
           org.opencv.objdetect.CascadeClassifier
           java.awt.image.BufferedImage))

(def FACE-XML "resources/haarcascade_frontalface_default.xml")

(def EYE-XML "resources/haarcascade_eye.xml")

(def RED (Scalar. 0.0  0.0 255.0))

(def GREEN (Scalar. 0.0 255.0 0.0))

(def BLUE (Scalar. 255.0 0.0 0.0))

(defn face-classifier
  "Creates an instance of an openCV classifier that will detect faces"
  []
  (CascadeClassifier. FACE-XML))

(defn eye-classifier
  "Creates an instance of an openCV classifier that will detect
  faces. Should be applied to a region of interest which is known to
  contain a face"
  []
  (CascadeClassifier. EYE-XML))

(defn read-image
  "Create an openCV matrix (Mat) from an image file"
  [filename]
  (Imgcodecs/imread filename))

(defn write-image
  "Write an openCV matric holding image data to a file in one of the
  supported image formats (PNG & JPEG)"
  [filename img]
  (Imgcodecs/imwrite filename img))

(defn result-matrix
  "Returns an openCV matrix with the same dimensions and type as the
  input matrix. Useful since most openCV functions that don't modify a
  matrix in-place expect a destination matrix to be supplied"
  [img]
  (Mat. (.rows img) (.cols img) (.type img)))

(defn to-grayscale
  "Convert an openCV matrix representing a colour image to a grayscale
  one of the same size"
  [img]
  (let [gray (Mat. (.rows img) (.cols img) CvType/CV_8UC3)]
    (Imgproc/cvtColor img gray Imgproc/COLOR_BGR2GRAY)
    gray))

(defn to-hsv
  "Convert an openCV matrix representing a colour image to a grayscale
  one of the same size"
  [img]
  (let [hsv (result-matrix img)]
    (Imgproc/cvtColor img hsv Imgproc/COLOR_BGR2HSV)
    hsv))

(defn to-java
  "Converts an openCV matrix representing an image to a Java BufferedImage"
  [img]
  (let [w (.cols img)
        h (.rows img)
        d (.elemSize img)
        t (if (= d 3)
            BufferedImage/TYPE_3BYTE_BGR
            BufferedImage/TYPE_BYTE_GRAY)
        bytes (make-array Byte/TYPE (int (* w h d)))
        image (BufferedImage. w h t)]
    (.get img 0 0 bytes)
    (-> image
        (.getRaster)
        (.setDataElements 0 0 w h bytes))
    image))

(defn make-gray-cell
  "Return a value capable of holding a single cell in a grayscalw image"
  []
  (make-array Byte/TYPE 1))

(defn from-gray-cell
  "Exract value from a gray cell"
  [cell]
  (Byte/toUnsignedInt (first cell)))

; TODO make capable of handling multi-channel images
(defn get-cell
  "Return the value of the cell at row, col in the grayscale image"
  ([img row col]
   (get-cell img row col (make-gray-cell)))
  ([img row col cell]
   (do
     (.get img row col cell)
     (from-gray-cell cell))))

; TODO make capable of handling multi-channel images
(defn get-row
  "Get a row from a grayscale image"
  [img row]
  (let [cell (make-gray-cell)]
    (map (fn [i] (get-cell img row i cell)) (range 0 (.cols img)))))

; TODO make capable of handling multi-channel images
(defn get-col
  "Get a column from a grayscale image"
  [img col]
  (let [cell (make-gray-cell)]
    (map (fn [i] (get-cell img i col cell)) (range 0 (.rows img)))))

(defn argmax
  "Return the index in the sequence of the cell with the max value"
  [vals]
  (.indexOf  vals (apply max vals)))

(defn argmax-row
  "For each row in the image return the index of the highest value"
  [img]
  (map (fn [i] (argmax (get-row img i))) (range 0 (.rows img))))

(defn resize-by-width
  "Resize an image by supplied the desired width"
  [img new-width]
  (let [w (.cols img)
        h (.rows img)
        ratio (/ new-width w)
        new-height (int (* h ratio))
        result (Mat. new-height new-width (.type img))]
    (Imgproc/resize  img result (Size. new-width new-height))
    result))

(defn resize-by-height
  "Resize an image by supplying the desired height"
  [img new-height]
    (let [w (.cols img)
        h (.rows img)
        ratio (/ new-height h)
        new-width (int (* w ratio))
        result (Mat. new-height new-width (.type img))]
    (Imgproc/resize  img result (Size. new-width new-height))
    result))

(defn gaussian-blur
  "Apply gaussian blurring to the supplied image. sigma X & sigma Y are
  calculated from the kernel size"
  [img kernel-size]
  (let [result (result-matrix img)]
    (Imgproc/GaussianBlur img result (Size. kernel-size kernel-size) 0.0)
    result))

(defn range-mask
  "Return a mask representing the pixels within the specified range"
  [img low high]
  (let [result (Mat. (.rows img) (.cols img) CvType/CV_8UC3)]
    (Core/inRange img low high result)
    result))

(defn erode
  "Erode image http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html"
  ([img] (erode img 3 1))
  ([img kernel-size iterations]
     (let [result (result-matrix img)
           se (Imgproc/getStructuringElement
               Imgproc/MORPH_ELLIPSE (Size. kernel-size kernel-size))]
       (Imgproc/erode  img result se (Point. -1 -1) iterations)
       result)))

(defn dilate
  "Dilate image http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html"
  ([img] (dilate img 3 1))
  ([img kernel-size iterations]
     (let [result (result-matrix img)
           se (Imgproc/getStructuringElement
               Imgproc/MORPH_ELLIPSE (Size. kernel-size kernel-size))]
       (Imgproc/dilate  img result se (Point. -1 -1) iterations)
       result)))

(defn matrix-variance
  "Return the variance of a single channel image matrix"
  [img]
  (let [mean (MatOfDouble.)
        stddev (MatOfDouble.)]
    (Core/meanStdDev img mean stddev)
    (let [sd (first (.toList stddev))]
      (* sd sd))))

(defn apply-classifier [clr img]
  (let [result (MatOfRect.)]
    (.detectMultiScale clr img result)
    (.toList result)))

(defn roi
  "Create a matrix representing the region of interest of a larger image
  defined by an openCV rect"
  [img rect]
  (.submat rect))

(defn min-point
  "Returns the corner of a rectanges with the smallest x & y values"
  [rect]
  (Point. (.x rect) (.y rect)))

(defn max-point
  "Returns the corner of a rectanges with the largest x & y values"
  [rect]
  (Point. (+ (.x rect) (.width rect)) (+ (.y rect) (.height rect))))

(defn offset-rect
  "Returns absolute version of rectanged embedded in larger
  rectangle. use this to convert a rectangle found by applying a
  classifier to a ROI into coordinates relative to the original image"
  [main embedded]
  (Rect. (+ (.x main) (.x embedded))
         (+ (.y main) (.y embedded))
         (.width embedded)
         (.height embedded)))

(defn apply-classifier-to-roi
  "Apply a classifier to the region of an image (matrix) defined by an
  openCv rect"
  [clr img rect]
  (let [result (MatOfRect.)
        roi (roi img rect)]
    (.detectMultiScale clr roi result)
    (.toList result)))

(defn faces-and-eyes
  "Find faces and eyes within those faces in the supplied grayscale image"
  [img]
  (let [faces (apply-classifier (face-classifier) img)
        eyes-in-faces
        (map #(apply-classifier-to-roi (eye-classifier) img %) faces)]
    (map (fn [f e] {:face f, :eyes e}) faces eyes-in-faces)))


(defn is-image-blurry
  "Determine if an image is blurry using the variance of a Laplacian of
  the grayscale image. From
  http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/"
  ([img]
     (is-image-blurry img 100))
  ([img threshold]
     (let [gray (to-grayscale img)
           laplacian (result-matrix gray)]
       (Imgproc/Laplacian gray laplacian CvType/CV_64F)
       (let [v (matrix-variance laplacian)]
         (< v threshold)))))

(defn hsv-mask
  "Convert an image into a mask indicating the pixels containing values
  within the supplied lower and upper bounds in HSV colour space"
  [img low high]
  (-> img
      (gaussian-blur 11)
      (to-hsv)
      (range-mask low high)
      (erode 3 2)
      (dilate 3 2)))

(defn find-contours
  "Find the contours in an image which is assumed to be grayscale
  http://docs.opencv.org/3.0.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a"
  [img]
  (let [contours (java.util.ArrayList.)]
    (Imgproc/findContours img contours (Mat.)
                          Imgproc/RETR_EXTERNAL
                          Imgproc/CHAIN_APPROX_SIMPLE)
    contours))

(defn largest-contour
  "Return the contour with the largest area"
  [contours]
  (if (seq contours)
    (apply max-key #(Imgproc/contourArea %) contours)
    nil))

(defn min-enclosing-circle
  "Given a contour return a map containing the X & Y coordinates of the
  centre and the radius"
  [contour]
  (let [centre (Point.)
        radius-array (make-array Float/TYPE 1)
        m2f (MatOfPoint2f.)]
    ; need to convert the contour from a MatOfPoint to MatOfPoint2f
    (.fromList m2f (.toList contour))
    ; now actually get the enclosing circle
    (Imgproc/minEnclosingCircle m2f centre radius-array)
    {:x (.x centre) :y (.y centre) :radius (first radius-array)}))

(defn find-blob
  "Return the centre and radius of the largest blob (if any) which lies
  within the range low-high in HSV colour space. Adapted from python
  implementation described in
  http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
  Ideally we would compute the centre using moments (as in the python
  example but the Moments class seems to have disapeared from openCV
  3.0.0, see github issue https://github.com/Itseez/opencv/issues/5017"
  [img low high]
  (if-let [contour (-> img
                       (hsv-mask low high)
                       (find-contours)
                       (largest-contour))]
    (assoc (min-enclosing-circle contour)
      :contour contour)))
