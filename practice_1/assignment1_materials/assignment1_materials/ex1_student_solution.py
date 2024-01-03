"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        """INSERT YOUR CODE HERE"""
        ones_vec = np.ones((1,match_p_dst.shape[1]))
        zeros_vec = np.zeros((3,match_p_dst.shape[1]))
        basic_vec = np.concatenate((match_p_src,ones_vec))
        minus_u_dst = -np.tile(match_p_dst[0,:],(3,1))
        minus_v_dst = -np.tile(match_p_dst[1,:],(3,1))
        a_us = np.concatenate((basic_vec,zeros_vec,minus_u_dst*basic_vec))
        a_vs = np.concatenate((zeros_vec,basic_vec,minus_v_dst*basic_vec))
        A = np.zeros((a_us.shape[0],a_us.shape[1]*2))
        A[:,0::2] = a_us
        A[:,1::2] = a_vs
        U, S, Vh = np.linalg.svd(A.T)
        h = np.reshape(Vh[-1,:],(3,3))
        
        return h

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        forward_map = np.zeros(dst_image_shape).astype(int)
        src_image_shape = src_image.shape
        for row in range(src_image_shape[0]):
            for column in range(src_image_shape[1]):
                pixel = src_image[row,column,:]
                coord_vec = np.ones((3,1))
                coord_vec[0] = column
                coord_vec[1] = row
                new_coord_vec = np.dot(homography,coord_vec)
                new_column = int(new_coord_vec[0]/new_coord_vec[2])
                new_row = int(new_coord_vec[1]/new_coord_vec[2])
                if 0 <= new_row and new_row < dst_image_shape[0] and 0 <= new_column and new_column < dst_image_shape[1]:
                    forward_map[new_row,new_column,:] = pixel
        return forward_map

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        forward_map = np.zeros(dst_image_shape).astype(int)

        # (1) Create a meshgrid of columns and rows
        src_image_shape = src_image.shape
        x_source =  np.linspace(0,src_image_shape[0]-1,src_image_shape[0]).astype(int)
        y_source = np.linspace(0,src_image_shape[1]-1,src_image_shape[1]).astype(int)
        meshgrid_source_x,meshgrid_source_y = np.meshgrid(x_source,y_source)
        # (2) pixel locations of homogeneous coordinates
        pixel_locations_homogeneous = np.zeros((3,src_image_shape[1],src_image_shape[0]))
        pixel_locations_homogeneous[0,:,:] = meshgrid_source_y
        pixel_locations_homogeneous[1,:,:] = meshgrid_source_x
        pixel_locations_homogeneous[2,:,:] = np.ones((src_image_shape[1],src_image_shape[0]))
        # (3) Transform the source homogeneous coordinates to the target
        dest_pixel_locations = np.tensordot(homography, pixel_locations_homogeneous, axes=(1, 0))
        #Apply the normalization you've seen in class.
        # (4) coordinates into integer values
        dest_pixel_locations = np.divide(dest_pixel_locations[0:2, :, :], dest_pixel_locations[2, :, :]).astype(int)
        # clip them according to the destination image size.
        cond_0_I = dest_pixel_locations[0] < dst_image_shape[1]
        cond_0_II = 0 <= dest_pixel_locations[0] 
        cond_1_I = dest_pixel_locations[1] < dst_image_shape[0]
        cond_1_II = 0 <= dest_pixel_locations[1] 
        cond_0 = cond_0_I*cond_0_II
        cond_1 = cond_1_I*cond_1_II
        final_cond = cond_0*cond_1
        final_cond_coord = np.nonzero(final_cond)
        dist_idx = dest_pixel_locations[:,final_cond_coord[0],final_cond_coord[1]]
        src_idx = pixel_locations_homogeneous[:,final_cond_coord[0],final_cond_coord[1]]
        #(5) Plant the pixels from the source image to the target image 
        forward_map[dist_idx[1].astype(int),dist_idx[0].astype(int)] = src_image[src_idx[1].astype(int),src_idx[0].astype(int)]
        
        return forward_map


    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
        fit_percent = 0
        dist_mse = 10 ** 9
        match_p_src_shape = match_p_src.shape
        match_p_src_homogeneous = np.ones((3,match_p_src.shape[1]))
        match_p_src_homogeneous[:2] = match_p_src
        match_p_dst_homogeneous = np.dot(homography,match_p_src_homogeneous)
        match_p_dst_homogeneous = np.divide(match_p_dst_homogeneous[0:2, :], match_p_dst_homogeneous[2, :]).astype(int)
        distance_matrix = np.sqrt(np.power(match_p_dst_homogeneous - match_p_dst,2).sum(axis=0))
        under_dist_th = distance_matrix < max_err
        inliers_num = np.sum(under_dist_th*1)
        fit_percent = inliers_num/match_p_src.shape[1]
        if not fit_percent==0:
            dist_mse = np.sum(np.power(distance_matrix[under_dist_th],2))/inliers_num   

        return (fit_percent,dist_mse)

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        match_p_src_shape = match_p_src.shape
        match_p_src_homogeneous = np.ones((3,match_p_src.shape[1]))
        match_p_src_homogeneous[:2] = match_p_src
        match_p_dst_homogeneous = np.dot(homography,match_p_src_homogeneous)
        match_p_dst_homogeneous = np.divide(match_p_dst_homogeneous[0:2, :], match_p_dst_homogeneous[2, :]).astype(int)
        distance_matrix = np.sqrt(np.power(match_p_dst_homogeneous - match_p_dst,2).sum(axis=0))
        under_dist_th = distance_matrix < max_err
        return match_p_src[:,under_dist_th],match_p_dst[:,under_dist_th]

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        w = inliers_percent
        # # t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        p = 0.99
        # # the minimal probability of points which meets with the model
        d = 0.5
        # # number of points sufficient to compute the model
        n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        t = max_err
        all_known_points = np.arange(match_p_src.shape[1])
        mse = 10**9
        best_homography = None
        for i in range(k):
            np.random.shuffle(all_known_points)
            random_points = all_known_points[:n]
            match_p_src_new, match_p_dst_new = match_p_src[:,random_points],match_p_dst[:,random_points]
            new_homography = self.compute_homography_naive(match_p_src_new,match_p_dst_new)
            #How many inliers do we have when taking random points?
            inliers_p_new_homography ,d_mse_new_homography = self.test_homography(new_homography,match_p_src,match_p_dst,t)
            #Meaning: model was proved because minimal number of points.
            if inliers_p_new_homography >= d:
                #Taking inliers from all points
                inliers_to_new_homography_p_src,inliers_to_new_homography_p_dst = self.meet_the_model_points(new_homography, match_p_src, match_p_dst, t)
                #New homography using inliers:
                homography_all_inliers = self.compute_homography_naive(inliers_to_new_homography_p_src,inliers_to_new_homography_p_dst)
                inliers_homography_all_inliers ,d_mse_homography_all_inliers = self.test_homography(homography_all_inliers,match_p_src,match_p_dst,t)
                if d_mse_homography_all_inliers < mse:
                    mse = d_mse_homography_all_inliers
                    best_homography = homography_all_inliers                   
        return best_homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        #(1) Create a mesh-grid of columns and rows of the destination image.
        x_dst =  np.linspace(0,dst_image_shape[0]-1,dst_image_shape[0]).astype(int)
        y_dst = np.linspace(0,dst_image_shape[1]-1,dst_image_shape[1]).astype(int)
        meshgrid_dst_x,meshgrid_dst_y = np.meshgrid(x_dst,y_dst)
        #(2) Create a set of homogenous coordinates for the destination image
        #using the mesh-grid from (1).
        pixel_locations_homogeneous_dst = np.zeros((3,dst_image_shape[1],dst_image_shape[0]))
        pixel_locations_homogeneous_dst[0,:,:] = meshgrid_dst_y
        pixel_locations_homogeneous_dst[1,:,:] = meshgrid_dst_x
        pixel_locations_homogeneous_dst[2,:,:] = np.ones((dst_image_shape[1],dst_image_shape[0]))
        #(3) Compute the corresponding coordinates in the source image using
        #the backward projective homography.
        src_pixel_locations = np.tensordot(backward_projective_homography, pixel_locations_homogeneous_dst, axes=(1, 0))
        src_pixel_locations = np.divide(src_pixel_locations[0:2, :, :], src_pixel_locations[2, :, :]).astype(int)
        #clipping src locations
        src_image_shape = src_image.shape
        cond_0_I = src_pixel_locations[0] < src_image_shape[1]
        cond_0_II = 0 <= src_pixel_locations[0] 
        cond_1_I = src_pixel_locations[1] < src_image_shape[0]
        cond_1_II = 0 <= src_pixel_locations[1] 
        cond_0 = cond_0_I*cond_0_II
        cond_1 = cond_1_I*cond_1_II
        final_cond = cond_0*cond_1
        final_cond_coord = np.nonzero(final_cond)
        src_idx = src_pixel_locations[:,final_cond_coord[0],final_cond_coord[1]]
        dst_idx = pixel_locations_homogeneous_dst[:,final_cond_coord[0],final_cond_coord[1]].astype(int)
        #(4) Create the mesh-grid of source image coordinates.
        x_src =  np.linspace(0,src_image_shape[0]-1,src_image_shape[0]).astype(int)
        y_src = np.linspace(0,src_image_shape[1]-1,src_image_shape[1]).astype(int)
        meshgrid_src_x,meshgrid_src_y = np.meshgrid(x_src,y_src)
        #(5) For each color channel (RGB): Use scipy's interpolation.griddata
        #with an appropriate configuration to compute the bi-cubic
        #interpolation of the projected coordinates.
        backward_map = np.zeros(dst_image_shape).astype(int)
        rgb = src_image[meshgrid_src_x.flatten(),meshgrid_src_y.flatten()]
        backward_map[dst_idx[1], dst_idx[0], :] = np.clip(griddata((meshgrid_src_y.flatten(),meshgrid_src_x.flatten()),rgb ,(src_idx[0],src_idx[1]),method='cubic'),0,255)

        return backward_map

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        #(1) Build the translation matrix from the pads.
        translation_matrix = np.array([[1,0,-pad_left],[0,1,-pad_up],[0,0,1]])
        #(2) Compose the backward homography and the translation matrix together.
        backward_homography_with_translation = np.dot(backward_homography,translation_matrix)
        #(3) Scale the homography as learnt in class.
        backward_homography_with_translation /= np.linalg.norm(backward_homography_with_translation)
        return backward_homography_with_translation

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        #(1) Compute the forward homography
        homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        #...and the panorama shape.
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image, dst_image, homography)
        #(2) Compute the backward homography.
        backward_homography = self.compute_homography(match_p_dst, match_p_src, inliers_percent, max_err)
        #(3) Add the appropriate translation to the homography so that the
        #source image will plant in place.
        #(4) Compute the backward warping with the appropriate translation.
        backward_homography_with_translation = self.add_translation_to_backward_homography(backward_homography, pad_struct.pad_left, pad_struct.pad_up)
        #(5) Create the an empty panorama image and plant there the
        #destination image.
        img_panorama = np.zeros((panorama_rows_num, panorama_cols_num,3))
        img_panorama[pad_struct.pad_up:dst_image.shape[0] + pad_struct.pad_up ,pad_struct.pad_left:dst_image.shape[1] + pad_struct.pad_left, :] = dst_image
        #(6) place the backward warped image in the indices where the panorama
        #image is zero.
        img_panorama = np.where(img_panorama == 0,self.compute_backward_mapping(backward_homography_with_translation, src_image, (panorama_rows_num, panorama_cols_num,3)),img_panorama)
        #(7) Don't forget to clip the values of the image to [0, 255].
        return np.clip(img_panorama, 0, 255).astype(np.uint8)

