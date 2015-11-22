(ns cljex-opencv.core
    (:import [org.opencv.core Core Point Rect Mat MatOfRect MatOfDouble CvType Size Scalar]
           org.opencv.imgcodecs.Imgcodecs
           org.opencv.imgproc.Imgproc
           org.opencv.objdetect.CascadeClassifier))

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

(defn colour-to-grayscale
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
     (let [gray (colour-to-grayscale img)
           laplacian (result-matrix gray)]
       (Imgproc/Laplacian gray laplacian CvType/CV_64F)
       (let [v (matrix-variance laplacian)]
         (< v threshold)))))
