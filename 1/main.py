# Kidd Pham
# colorize_skel.py
# 12 September 2025

import numpy as np
import skimage.io as skio
import skimage as sk
import imageio.v3 as iio
from skimage.color import rgb2gray
from skimage.transform import resize

def loadAndSplit(imagePath) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = skio.imread(imagePath)

    image = sk.img_as_float(image).astype(np.float32, copy=False)
    if image.ndim == 3:
        image = rgb2gray(image).astype(np.float32, copy=False)
    image = trimImageBorders(image)

    #height = im.shape[0] // 3
    #b = im[:height]
    #g = im[height:2*height]
    #r = im[2*height:3*height]
    height = (image.shape[0] // 3) * 3
    image = image[:height]
    blue, green, red = np.array_split(image, 3, axis=0)
    
    return blue, green, red

#range-based trimming
""" 
def trim_image_borders(image) -> np.ndarray:
    height, width = image.shape
    MAX_Y = int(0.15 * height)
    MAX_X = int(0.15 * width)
    
    def topL_nonflat(arr, limit):
        tolerance = 0.02
        for i in range(limit):
            if arr[i] >= tolerance:
                return i
        return limit
    def botR_nonflat(arr, limit):
        tolerance = 0.02
        for i in range(limit):
            if arr[-i - 1] >= tolerance:
                return i
        return limit
    
    rowRange = image.max(axis=1) - image.min(axis=1)
    colRange = image.max(axis=0) - image.min(axis=0)
    topEdge = topL_nonflat(rowRange, MAX_Y)
    botEdge = botR_nonflat(rowRange, MAX_Y)
    leftEdge = topL_nonflat(colRange, MAX_X)
    rightEdge = botR_nonflat(colRange, MAX_X)

    return image[topEdge:height - botEdge, leftEdge:width - rightEdge]
"""

#std-based trimming
def trimImageBorders(image) -> np.ndarray:
    height, width = image.shape
    maxY = int(0.05 * height)  # Even less trimming
    maxX = int(0.05 * width)

    # Cut out 10% of images on all sides then compute the std of each each row/column
    sideMargin = max(1, int(0.02 * width))  # Minimal margins
    rowStd = image[:, sideMargin:width - sideMargin].std(axis=1)
    topBottomMargin = max(1, int(0.02 * height))
    colStd = image[topBottomMargin:height - topBottomMargin].std(axis=0)

    # Cut out a middle portion of the image and compute the median of the std in that portion
    # Then, set the threshold as 0.25 of each median. This will be used to locate "edges"
    rowMid = rowStd[height // 4:3 * height // 4]
    colMid = colStd[width // 4:3 * width // 4]
    thresholdRow = 0.5 * np.median(rowMid)  # Higher threshold = less trimming
    thresholdCol = 0.5 * np.median(colMid)

    # Scan from top, bottom, left, right to find edges, looking for a run of 3 streak of above the threshold
    # Counts above threshold if the std of that specific array is above the threshold
    contentBorder = 1  # Shorter run requirement
    def scanForward(arr, limit, thr) -> int:
        run = 0
        for i in range(min(limit, arr.size)):
            run = run + 1 if arr[i] > thr else 0
            if run >= contentBorder:
                return max(0, i - contentBorder + 1)
        return min(limit, arr.size)

    def scanBackward(arr, limit, thr) -> int:
        run = 0
        for i in range(min(limit, arr.size)):
            run = run + 1 if arr[-1 - i] > thr else 0
            if run >= contentBorder:
                return max(0, i - contentBorder + 1)
        return min(limit, arr.size)

    # plug n chug
    topEdge = scanForward(rowStd, maxY, thresholdRow)
    bottomEdge = scanBackward(rowStd, maxY, thresholdRow)
    leftEdge = scanForward(colStd, maxX, thresholdCol)
    rightEdge = scanBackward(colStd, maxX, thresholdCol)

    return image[topEdge:height - bottomEdge, leftEdge:width - rightEdge]

def bestShift(referenceImage, currentImage) -> tuple[int, int]:
    maxDisplacement = 15
    
    # Very minimal processing for small images
    if min(referenceImage.shape) < 500:
        cropFraction = 0.05  # Almost no cropping - use 90% of image
        do_subsample = False  # No subsampling for small images
    else:
        cropFraction = 0.25  # Light cropping for larger images
        do_subsample = True

    def centralCropFraction(im, frac) -> np.ndarray:
        height, width = im.shape
        dh = int(height * frac); dw = int(width * frac)
        if dh == 0 or dw == 0:
            return im
        return im[dh:height - dh, dw:width - dw]

    def overlapViews(a, b, dy, dx) -> tuple[np.ndarray | None, np.ndarray | None]:
        h, w = a.shape
        y0a = max(0, dy)
        y0b = max(0, -dy)
        x0a = max(0, dx)
        x0b = max(0, -dx)
        heightOverlap = min(h - y0a, h - y0b)
        widthOverlap  = min(w - x0a, w - x0b)
        if heightOverlap <= 0 or widthOverlap <= 0:
            return None, None
        return a[y0a:y0a + heightOverlap, x0a:x0a + widthOverlap], b[y0b:y0b + heightOverlap, x0b:x0b + widthOverlap]

    # 1time crop
    refCropped = centralCropFraction(referenceImage.astype(np.float32, copy=False), cropFraction)
    curCropped = centralCropFraction(currentImage.astype(np.float32, copy=False), cropFraction)

    # subsample once (2x) for scoring — lebron level speed win, ben level quality loss
    if do_subsample:
        refCropped = refCropped[::2, ::2]
        curCropped = curCropped[::2, ::2]

    # brute force search and take best ncc 
    def nccScore(a: np.ndarray, b: np.ndarray) -> float:
        aZero = a - a.mean()
        bZero = b - b.mean()
        na = np.linalg.norm(aZero)
        nb = np.linalg.norm(bZero)
        if na == 0.0 or nb == 0.0:
            return -np.inf
        return float((aZero * bZero).sum() / (na * nb))

    bestDy, bestDx, bestScore = 0, 0, -np.inf
    for rowOffset in range(-maxDisplacement, maxDisplacement + 1):
        for colOffset in range(-maxDisplacement, maxDisplacement + 1):
            viewRef, viewCur = overlapViews(refCropped, curCropped, rowOffset, colOffset)
            if viewRef is None:
                continue
            score = nccScore(viewRef, viewCur)
            if score > bestScore:
                bestDy, bestDx, bestScore = rowOffset, colOffset, score
    return bestDy, bestDx

# clean the edges after gigachad alignment
def cropOverlap(blue, green, red, shiftGreen, shiftRed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = blue.shape
    greenRowChange, greenColChange = shiftGreen
    redRowChange, redColChange = shiftRed

    topEdge = max(0, greenRowChange, redRowChange)
    bottomEdge = min(height, height + min(greenRowChange, redRowChange))
    leftEdge = max(0, greenColChange, redColChange)
    rightEdge = min(width, width + min(greenColChange, redColChange))

    newBlue = blue[topEdge:bottomEdge, leftEdge:rightEdge]
    
    if greenRowChange != 0 or greenColChange != 0:
        newGreen = np.roll(green, (greenRowChange, greenColChange), (0, 1))[topEdge:bottomEdge, leftEdge:rightEdge]
    else:
        newGreen = green[topEdge:bottomEdge, leftEdge:rightEdge]
        
    if redRowChange != 0 or redColChange != 0:
        newRed = np.roll(red, (redRowChange, redColChange), (0, 1))[topEdge:bottomEdge, leftEdge:rightEdge]
    else:
        newRed = red[topEdge:bottomEdge, leftEdge:rightEdge]

    return newBlue, newGreen, newRed

# plug n chug
def alignSingleScale(imagePath, outputPath) -> None:
    blue, green, red = loadAndSplit(imagePath)
    greenDy, greenDx = bestShift(blue, green)
    redDy, redDx   = bestShift(blue, red)
    newBlue, newGreen, newRed = cropOverlap(blue, green, red, (greenDy, greenDx), (redDy, redDx))
    rgb = np.dstack([newRed, newGreen, newBlue])
    iio.imwrite(outputPath, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    print("Image G shifted by : ", (greenDy, greenDx), "Image R shifted by :", (redDy, redDx))

#alignSingleScale("cs180_proj1_data/cathedral.jpg", "cathedral.jpg")
#alignSingleScale("cs180_proj1_data/monastery.jpg", "monastery_aligned.jpg")
#alignSingleScale("cs180_proj1_data/tobolsk.jpg", "tobolsk_aligned.jpg")

def bestShiftPyramid(referenceImage, currentImage) -> tuple[int, int]:
    # Build proper multi-level pyramid
    ref = referenceImage.astype(np.float32, copy=False)
    img = currentImage.astype(np.float32, copy=False)
    
    # Build pyramid levels using recursive downsampling
    refs = [ref]
    imgs = [img]
    
    # Create multiple levels until image is small enough
    while min(refs[-1].shape) > 100:
        # Downsample by factor of 2
        prev_ref = refs[-1]
        prev_img = imgs[-1]
        
        new_ref = prev_ref[::2, ::2]
        new_img = prev_img[::2, ::2]
        
        refs.append(new_ref)
        imgs.append(new_img)
    
    # Start at coarsest level (last in list)
    coarse_level = len(refs) - 1
    
    # Initial search at coarsest level with large window
    def searchAtLevel(ref_level, img_level, center_dy=0, center_dx=0, search_radius=15):
        cropFraction = 0.3
        
        # Crop images to focus on center content
        def centralCropFraction(im, frac) -> np.ndarray:
            height, width = im.shape
            dh = int(height * frac); dw = int(width * frac)
            if dh == 0 or dw == 0:
                return im
            return im[dh:height - dh, dw:width - dw]

        def overlapViews(a, b, dy, dx) -> tuple[np.ndarray | None, np.ndarray | None]:
            h, w = a.shape
            y0a = max(0, dy)
            y0b = max(0, -dy)
            x0a = max(0, dx)
            x0b = max(0, -dx)
            heightOverlap = min(h - y0a, h - y0b)
            widthOverlap = min(w - x0a, w - x0b)
            if heightOverlap <= 0 or widthOverlap <= 0:
                return None, None
            return a[y0a:y0a + heightOverlap, x0a:x0a + widthOverlap], b[y0b:y0b + heightOverlap, x0b:x0b + widthOverlap]

        refCropped = centralCropFraction(ref_level, cropFraction)
        curCropped = centralCropFraction(img_level, cropFraction)

        def nccScore(a: np.ndarray, b: np.ndarray) -> float:
            aZero = a - a.mean()
            bZero = b - b.mean()
            na = np.linalg.norm(aZero)
            nb = np.linalg.norm(bZero)
            if na == 0.0 or nb == 0.0:
                return -np.inf
            return float((aZero * bZero).sum() / (na * nb))

        bestDy, bestDx, bestScore = center_dy, center_dx, -np.inf
        
        # Search in window around center
        for rowOffset in range(center_dy - search_radius, center_dy + search_radius + 1):
            for colOffset in range(center_dx - search_radius, center_dx + search_radius + 1):
                viewRef, viewCur = overlapViews(refCropped, curCropped, rowOffset, colOffset)
                if viewRef is None:
                    continue
                score = nccScore(viewRef, viewCur)
                if score > bestScore:
                    bestDy, bestDx, bestScore = rowOffset, colOffset, score
        
        return bestDy, bestDx
    
    # Search at coarsest level with large window
    dy, dx = searchAtLevel(refs[coarse_level], imgs[coarse_level], 0, 0, 15)
    
    # Refine through pyramid levels
    for level in range(coarse_level - 1, -1, -1):
        # Scale up the displacement estimate
        dy *= 2
        dx *= 2
        
        # Refine with smaller search window around scaled estimate
        search_radius = 3 if level > 0 else 2
        dy, dx = searchAtLevel(refs[level], imgs[level], dy, dx, search_radius)
    
    return dy, dx

def alignPyramid(imagePath, outputPath) -> None:
    import time
    start = time.time()
    
    blue, green, red = loadAndSplit(imagePath)
    print(f"Image size: {blue.shape}, Load time: {time.time() - start:.2f}s")
    
    t1 = time.time()
    greenDy, greenDx = bestShiftPyramid(blue, green)
    print(f"Green alignment: {time.time() - t1:.2f}s")
    
    t2 = time.time()
    redDy, redDx = bestShiftPyramid(blue, red)
    print(f"Red alignment: {time.time() - t2:.2f}s")
    
    newBlue, newGreen, newRed = cropOverlap(blue, green, red, (greenDy, greenDx), (redDy, redDx))
    rgb = np.dstack([newRed, newGreen, newBlue])
    iio.imwrite(outputPath, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    print(f"Total time: {time.time() - start:.2f}s")
    print("Image G shifted by :", (greenDy, greenDx), "Image R shifted by :", (redDy, redDx))


if __name__ == "__main__":
    alignPyramid("cs180_proj1_data/three_generations.tif", "three_generations_pyr.jpg")
    #alignPyramid("cs180_proj1_data/church.tif", "church_pyr.jpg")
    #alignPyramid("cs180_proj1_data/three_generations.tif", "three_generations.jpg")
    #alignSingleScale("cs180_proj1_data/tobolsk.jpg", "tobolsk_aligned.jpg")
    #alignSingleScale("cs180_proj1_data/cathedral.jpg", "cathedral.jpg")
