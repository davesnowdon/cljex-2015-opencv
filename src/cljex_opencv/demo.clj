(ns cljex-opencv.demo
  (:require [cljex-opencv.core :refer :all])
  (:import [org.opencv.core Core Point Rect Mat MatOfRect MatOfDouble CvType Size Scalar]
           org.opencv.imgcodecs.Imgcodecs
           org.opencv.imgproc.Imgproc
           org.opencv.objdetect.CascadeClassifier))

(defn show-faces-and-eyes
  "Demonstration function showing how to make use of the values returned
  by an openCV classified by drawing rectangles around them"
  ([img]
     (show-faces-and-eyes img BLUE GREEN))
  ([img face-outline-colour eye-outline-colour]
     (let [gray (colour-to-grayscale img)
           faces-and-eyes (faces-and-eyes gray)]
       (doseq [fe faces-and-eyes]
         (let [{:keys [face eyes]} fe
               eye-rects (map #(offset-rect face %) eyes)]
           (Imgproc/rectangle img
                              (min-point face) (max-point face)
                              face-outline-colour 2)
           (doseq [e eye-rects]
             (Imgproc/rectangle img
                                (min-point e) (max-point e)
                                eye-outline-colour 2)))))))

(defn show-blob
  "Find a blob in the specified image and if it exists draw the outline
  and the centre"
  ([img low high]
     (show-blob img low high BLUE GREEN))
  ([img low high centre-colour outline-colour]
     (if-let [blob (find-blob img low high)]
       (do
         (Imgproc/circle img
                         (Point. (:x blob) (:y blob))
                         5
                         centre-colour
                         2)
         (Imgproc/drawContours img
                               (java.util.ArrayList. [(:contour blob)])
                               0
                               outline-colour
                               2)))))

(defn show-line
  "Attempt to detect a line in the image and draw the detected line over
  the original"
  ([img]
     (show-line img BLUE))
  ([img line-colour]
     (if-let [line (find-line img)]
       (let [[offset orientation] line
             width (.cols img)
             height (.rows img)
             half-height (/ height 2)
             x (+ half-height (* offset half-height))]
         (Imgproc/line img
                       (Point. x 0)
                       (Point. x (- height 1))
                       line-colour 2)))))
