package com.soundcloud.lsh

import org.apache.spark.mllib.linalg.Vector

/**
 * interface defining similarity measurement between 2 vectors
 */
trait VectorDistance extends Serializable {
  def apply(vecA: Vector, vecB: Vector): Double
}

/**
 * implementation of [[VectorDistance]] that computes cosine similarity
 * between two vectors
 */
object Cosine extends VectorDistance {

  def apply(vecA: Vector, vecB: Vector): Double = {
    val vecAarray = vecA.toArray
    val vecBarray = vecB.toArray
    dotProduct(vecAarray, vecBarray) / (l2(vecAarray) * l2(vecBarray))
  }

  def dotProduct(vecA: Array[Double], vecB: Array[Double]): Double = {
    vecA.view.zip(vecB.view).map { case (a, b) => a * b}.sum
  }

  def l2(vec: Array[Double]): Double = {
    math.sqrt(dotProduct(vec, vec))
  }

}

object Euclidean extends VectorDistance {
  def apply(vecA: Vector, vecB: Vector): Double = {
    math.sqrt(vecA.toArray.zip(vecB.toArray).map(p => p._1 - p._2).map(d => d * d).sum)
  }
}

object Manhattan extends VectorDistance {
  def apply(vecA: Vector, vecB: Vector): Double = {
    (vecA.toArray, vecB.toArray).zipped.foldLeft(0.0)((s, t) => s + math.abs(t._1 - t._2))
  }
}
