(ns cljex-opencv.util)

(defn mean [v] (/ (apply + v) (count v)))
