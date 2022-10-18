const Jimp = require('jimp')
const { cv } = require('opencv-wasm')

cv.idft = function (src, dst, flags, nonzero_rows) {
  cv.dft(src, dst, flags | cv.DFT_INVERSE, nonzero_rows)
}

function shiftDFT(mag) {
  const rect = new cv.Rect(0, 0, mag.cols & -2, mag.rows & -2)
  mag.roi(rect)

  const cx = mag.cols / 2
  const cy = mag.rows / 2

  const q0 = mag.roi(new cv.Rect(0, 0, cx, cy))
  const q1 = mag.roi(new cv.Rect(cx, 0, cx, cy))
  const q2 = mag.roi(new cv.Rect(0, cy, cx, cy))
  const q3 = mag.roi(new cv.Rect(cx, cy, cx, cy))

  const tmp = new cv.Mat()
  q0.copyTo(tmp)
  q3.copyTo(q0)
  tmp.copyTo(q3)

  q1.copyTo(tmp)
  q2.copyTo(q1)
  tmp.copyTo(q2)

  tmp.delete()
  q0.delete()
  q1.delete()
  q2.delete()
  q3.delete()
}

function getBlueChannel(image) {
  const channel = new cv.MatVector()
  cv.split(image, channel)
  const res = channel.get(0)
  channel.delete()
  return res
}

function getDftMat(padded) {
  const planes = new cv.MatVector()
  planes.push_back(padded)
  const matZ = new cv.Mat.zeros(padded.size(), cv.CV_32F)
  planes.push_back(matZ)
  const comImg = new cv.Mat()
  cv.merge(planes, comImg)
  cv.dft(comImg, comImg)
  matZ.delete()
  planes.delete()
  return comImg
}

function addTextByMat(comImg, watermarkText, point, fontSize) {
  cv.putText(
    comImg,
    watermarkText,
    point,
    cv.FONT_HERSHEY_DUPLEX,
    fontSize,
    cv.Scalar.all(0),
    2,
  )
  cv.flip(comImg, comImg, -1)
  cv.putText(
    comImg,
    watermarkText,
    point,
    cv.FONT_HERSHEY_DUPLEX,
    fontSize,
    cv.Scalar.all(0),
    2,
  )
  cv.flip(comImg, comImg, -1)
}

function transFormMatWithText(srcImg, watermarkText, fontSize) {
  const padded = getBlueChannel(srcImg)
  padded.convertTo(padded, cv.CV_32F)
  const comImg = getDftMat(padded)
  // add text
  const center = new cv.Point(padded.cols / 2, padded.rows / 2)
  addTextByMat(comImg, watermarkText, center, fontSize)
  const outer = new cv.Point(5, 30)
  addTextByMat(comImg, watermarkText, outer, fontSize)
  //back image
  const invDFT = new cv.Mat()
  cv.idft(comImg, invDFT, cv.DFT_SCALE | cv.DFT_REAL_OUTPUT, 0)
  const restoredImage = new cv.Mat()
  invDFT.convertTo(restoredImage, cv.CV_8U)
  const backPlanes = new cv.MatVector()
  cv.split(srcImg, backPlanes)
  // backPlanes.erase(backPlanes.get(0));
  // backPlanes.insert(backPlanes.get(0), restoredImage);
  backPlanes.set(0, restoredImage)
  const backImage = new cv.Mat()
  cv.merge(backPlanes, backImage)

  backPlanes.delete()
  padded.delete()
  comImg.delete()
  invDFT.delete()
  restoredImage.delete()
  return backImage
}

function getTextFormMat(backImage) {
  const padded = getBlueChannel(backImage)
  padded.convertTo(padded, cv.CV_32F)
  const comImg = getDftMat(padded)
  const backPlanes = new cv.MatVector()
  // split the comples image in two backPlanes
  cv.split(comImg, backPlanes)
  const mag = new cv.Mat()
  // compute the magnitude
  cv.magnitude(backPlanes.get(0), backPlanes.get(1), mag)
  // move to a logarithmic scale
  const matOne = cv.Mat.ones(mag.size(), cv.CV_32F)
  cv.add(matOne, mag, mag)
  cv.log(mag, mag)
  shiftDFT(mag)
  mag.convertTo(mag, cv.CV_8UC1)
  cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

  backPlanes.delete()
  padded.delete()
  comImg.delete()
  matOne.delete()
  return mag
}

function matToBuffer(mat) {
  if (!(mat instanceof cv.Mat)) {
    throw new Error('Please input the valid new cv.Mat instance.')
  }
  const img = new cv.Mat()
  const depth = mat.type() % 8
  const scale = depth <= cv.CV_8S ? 1 : depth <= cv.CV_32S ? 1 / 256 : 255
  const shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128 : 0
  mat.convertTo(img, cv.CV_8U, scale, shift)
  switch (img.type()) {
    case cv.CV_8UC1:
      cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA)
      break
    case cv.CV_8UC3:
      cv.cvtColor(img, img, cv.COLOR_RGB2RGBA)
      break
    case cv.CV_8UC4:
      break
    default:
      throw new Error(
        'Bad number of channels (Source image must have 1, 3 or 4 channels)',
      )
  }
  const imgData = Buffer.from(img.data)
  img.delete()
  return imgData
}

async function transformImageWithText(
  srcFileName,
  watermarkText,
  fontSize,
  enCodeFileName = '',
) {
  if (typeof srcFileName != 'string' && !(srcFileName instanceof Buffer)) {
    throw new Error('fileName must be string or Buffer')
  }
  if (typeof watermarkText != 'string') {
    throw new Error('waterMarkText must be string')
  }
  if (typeof fontSize != 'number') {
    throw new Error('fontSize must be number')
  }
  if (typeof enCodeFileName != 'string') {
    throw new Error('outFileName must be string')
  }
  const jimpSrc = await Jimp.read(srcFileName)
  const srcImg = new cv.matFromImageData(jimpSrc.bitmap)
  if (srcImg.empty()) {
    throw new Error('read image failed')
  }
  const comImg = transFormMatWithText(srcImg, watermarkText, fontSize)
  const imgRes = new Jimp({
    width: comImg.cols,
    height: comImg.rows,
    data: matToBuffer(comImg),
  })
  srcImg.delete()
  comImg.delete()
  if (enCodeFileName) {
    return await imgRes.writeAsync(enCodeFileName)
  } else {
    return imgRes
  }
}

async function getTextFormImage(enCodeFileName, deCodeFileName = '') {
  if (
    typeof enCodeFileName != 'string' &&
    !(enCodeFileName instanceof Buffer)
  ) {
    throw new Error('fileName must be string or Buffer')
  }
  if (typeof deCodeFileName != 'string') {
    throw new Error('backFileName must be string')
  }

  const jimpSrc = await Jimp.read(enCodeFileName)
  const comImg = new cv.matFromImageData(jimpSrc.bitmap)
  const backImage = getTextFormMat(comImg)
  const imgRes = await new Jimp({
    width: backImage.cols,
    height: backImage.rows,
    data: matToBuffer(backImage),
  })
  comImg.delete()
  backImage.delete()
  if (deCodeFileName) {
    return await imgRes.writeAsync(deCodeFileName)
  } else {
    return imgRes
  }
}

module.exports.transformImageWithText = transformImageWithText
module.exports.getTextFormImage = getTextFormImage
