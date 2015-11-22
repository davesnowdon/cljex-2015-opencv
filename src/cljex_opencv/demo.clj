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
