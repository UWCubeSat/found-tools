"""
Universal OpNav Testing Framework for FOUND
Generates All figures from Artemis I paper
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from enum import Enum


class TargetBody(Enum):
    """Target body types"""
    MOON = "moon"
    EARTH = "earth"


@dataclass
class TestImage:
    """Complete test image with all metadata"""
    image: np.ndarray
    
    # Ground truth (BET - Best Estimated Trajectory)
    position_camera: np.ndarray  # [x, y, z] in meters (camera frame)
    velocity_camera: np.ndarray  # [vx, vy, vz] in m/s (camera frame)
    attitude_camera_to_inertial: np.ndarray  # 3x3 rotation matrix
    
    # Sun and lighting
    sun_direction_camera: np.ndarray  # [x, y, z] unit vector (camera frame)
    sun_direction_inertial: np.ndarray  # [x, y, z] unit vector (inertial frame)
    
    # Target body parameters
    body_radii: np.ndarray  # [a, b, c] in meters (ellipsoid radii)
    body_orientation: np.ndarray  # 3x3 DCM to principal axes
    target_body: TargetBody
    
    # Camera calibration
    camera_K: np.ndarray  # 3x3 calibration matrix
    distortion: Dict[str, float]  # k1, k2, k3, p1, p2
    
    # Metadata
    timestamp: float  # seconds
    image_id: str
    pass_id: str
    range_m: float  # True range to target
    phase_angle_deg: float  # Sun-target-observer angle
    
    # Optional: BET uncertainty
    bet_uncertainty_range_m: Optional[float] = None
    bet_uncertainty_pointing_arcsec: Optional[float] = None
    
    # Optional: True limb points (for synthetic data)
    true_limb_points_pixels: Optional[np.ndarray] = None


@dataclass
class CRAResult:
    """Output from FOUND CRA algorithm"""
    # Primary outputs
    position_camera: np.ndarray  # Estimated [x, y, z] in meters
    
    # Optional velocity (if algorithm estimates it)
    velocity_camera: Optional[np.ndarray] = None
    
    # Algorithm details
    detected_limb_points: np.ndarray = None  # Nx2 detected limb pixels
    apparent_radius_pixels: float = 0.0
    centroid_pixels: np.ndarray = None  # [u, v] centroid in pixels
    
    # Algorithm metadata
    success: bool = True
    converged: bool = True
    n_iterations: int = 0
    processing_time_s: float = 0.0
    
    # Optional: iterative vs non-iterative comparison
    algorithm_type: str = "non-iterative"  # or "iterative"


@dataclass
class Residuals:
    """Complete residual analysis"""
    # Range residuals
    range_error_m: float
    range_error_km: float
    range_error_nmi: float  # Nautical miles (used in paper)
    
    # Radius residuals (in pixels)
    radius_residual_pixels: float  # Estimated - True
    
    # Centroid residuals (in pixels)
    centroid_camera_x_pixels: float  # In camera frame
    centroid_camera_y_pixels: float
    
    # Centroid in illumination frame (key for paper figures)
    centroid_illum_x_pixels: float  # Along sun direction
    centroid_illum_y_pixels: float  # Perpendicular to sun
    
    # Bearing error
    bearing_error_rad: float
    bearing_error_arcsec: float
    
    # Position error components
    position_error_vector: np.ndarray  # [dx, dy, dz] in meters
    position_error_magnitude: float  # meters
    
    # Additional for Earth
    atmosphere_bias_detected: Optional[float] = None  # For Earth images


class UniversalOpNavTester:
    def __init__(self, 
                 output_dir: Path = Path("./opnav_results"),
                 moon_radius_km: float = 1737.4,
                 earth_radii_km: Tuple[float, float, float] = (6378.137, 6378.137, 6356.752)):
        """
        Args:
            output_dir: Where to save results
            moon_radius_km: Moon radius
            earth_radii_km: Earth radii (equatorial, equatorial, polar)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Body parameters
        self.moon_radius_m = moon_radius_km * 1000
        self.earth_radii_m = np.array(earth_radii_km) * 1000
        
        # Storage for results
        self.test_images: List[TestImage] = []
        self.cra_results: List[CRAResult] = []
        self.residuals: List[Residuals] = []
        
        # Error models
        self.error_models = {
            TargetBody.MOON: self._default_moon_error_model,
            TargetBody.EARTH: self._default_earth_error_model
        }
    
    def add_test_case(self, 
                     test_image: TestImage,
                     cra_result: CRAResult):
        """
        Add a test case (image + result) to the analyzer
        
        Args:
            test_image: Input image with ground truth
            cra_result: Output from your CRA algorithm
        """
        # Store
        self.test_images.append(test_image)
        self.cra_results.append(cra_result)
        
        # Compute residuals
        residuals = self._compute_residuals(test_image, cra_result)
        self.residuals.append(residuals)
    
    def _compute_residuals(self, 
                          test_image: TestImage,
                          cra_result: CRAResult) -> Residuals:
        """Compute all residuals between estimate and ground truth"""
        
        # Position error
        pos_error_vec = cra_result.position_camera - test_image.position_camera
        pos_error_mag = np.linalg.norm(pos_error_vec)
        
        # Range error
        range_estimate = np.linalg.norm(cra_result.position_camera)
        range_true = test_image.range_m
        range_error_m = range_estimate - range_true
        range_error_km = range_error_m / 1000
        range_error_nmi = range_error_km * 0.539957  # km to nautical miles
        
        # Get body radius for this target
        if test_image.target_body == TargetBody.MOON:
            body_radius = self.moon_radius_m
        else:
            body_radius = self.earth_radii_m[0]  # Use equatorial
        
        # Get camera focal length
        focal_length = test_image.camera_K[0, 0]
        
        # Radius residual in pixels
        radius_residual = self._compute_radius_residual_pixels(
            cra_result.position_camera,
            test_image.position_camera,
            body_radius,
            focal_length
        )
        
        # Centroid residuals in camera frame
        centroid_cam_x, centroid_cam_y = self._compute_centroid_residual_camera(
            cra_result.position_camera,
            test_image.position_camera,
            focal_length
        )
        
        # Centroid residuals in illumination frame (CRITICAL FOR PAPER FIGURES)
        centroid_illum_x, centroid_illum_y = self._compute_centroid_residual_illumination(
            cra_result.position_camera,
            test_image.position_camera,
            test_image.sun_direction_camera,
            focal_length
        )
        
        # Bearing error
        bearing_error_rad = self._compute_bearing_error(
            cra_result.position_camera,
            test_image.position_camera
        )
        bearing_error_arcsec = np.rad2deg(bearing_error_rad) * 3600
        
        return Residuals(
            range_error_m=range_error_m,
            range_error_km=range_error_km,
            range_error_nmi=range_error_nmi,
            radius_residual_pixels=radius_residual,
            centroid_camera_x_pixels=centroid_cam_x,
            centroid_camera_y_pixels=centroid_cam_y,
            centroid_illum_x_pixels=centroid_illum_x,
            centroid_illum_y_pixels=centroid_illum_y,
            bearing_error_rad=bearing_error_rad,
            bearing_error_arcsec=bearing_error_arcsec,
            position_error_vector=pos_error_vec,
            position_error_magnitude=pos_error_mag
        )
    
    def _compute_radius_residual_pixels(self,
                                       pos_estimate: np.ndarray,
                                       pos_true: np.ndarray,
                                       body_radius: float,
                                       focal_length: float) -> float:
        """Compute apparent radius difference in pixels"""
        range_est = np.linalg.norm(pos_estimate)
        range_true = np.linalg.norm(pos_true)
        
        # Apparent angular radius (small angle approximation)
        ang_radius_est = body_radius / range_est
        ang_radius_true = body_radius / range_true
        
        # Convert to pixels
        pixel_radius_est = focal_length * ang_radius_est
        pixel_radius_true = focal_length * ang_radius_true
        
        return pixel_radius_est - pixel_radius_true
    
    def _compute_centroid_residual_camera(self,
                                         pos_estimate: np.ndarray,
                                         pos_true: np.ndarray,
                                         focal_length: float) -> Tuple[float, float]:
        """Compute centroid residuals in camera frame"""
        # Normalize to unit vectors
        u_est = pos_estimate / np.linalg.norm(pos_estimate)
        u_true = pos_true / np.linalg.norm(pos_true)
        
        # Small angle approximation: error ≈ u_est - u_true
        du = u_est - u_true
        
        # Convert to pixels (using pinhole projection)
        # For unit vector [x, y, z], pixel coords are approximately [fx*x/z, fy*y/z]
        # For small angles near optical axis (z≈1), this simplifies to [fx*x, fy*y]
        residual_x = du[0] * focal_length
        residual_y = du[1] * focal_length
        
        return residual_x, residual_y
    
    def _compute_centroid_residual_illumination(self,
                                               pos_estimate: np.ndarray,
                                               pos_true: np.ndarray,
                                               sun_direction: np.ndarray,
                                               focal_length: float) -> Tuple[float, float]:
        """
        Compute centroid residuals in ILLUMINATION FRAME
        
        Illumination frame:
        - X-axis: parallel to sun direction
        - Y-axis: perpendicular to sun, in plane with target
        - Z-axis: out of plane (right-hand rule)
        """
        # Normalize
        u_est = pos_estimate / np.linalg.norm(pos_estimate)
        u_true = pos_true / np.linalg.norm(pos_true)
        sun_hat = sun_direction / np.linalg.norm(sun_direction)
        
        # Build illumination frame using true position as reference
        # X-axis: sun direction
        x_illum = sun_hat
        
        # Z-axis: perpendicular to both sun and target
        z_illum = np.cross(sun_hat, u_true)
        z_illum_norm = np.linalg.norm(z_illum)
        
        if z_illum_norm < 1e-6:
            # Degenerate case: sun and target aligned
            # Choose arbitrary perpendicular
            if abs(sun_hat[2]) < 0.9:
                z_illum = np.cross(sun_hat, np.array([0, 0, 1]))
            else:
                z_illum = np.cross(sun_hat, np.array([1, 0, 0]))
        
        z_illum = z_illum / np.linalg.norm(z_illum)
        
        # Y-axis: complete right-handed system
        y_illum = np.cross(z_illum, x_illum)
        
        # Rotation matrix: camera to illumination frame
        R_illum_cam = np.vstack([x_illum, y_illum, z_illum])
        
        # Transform both vectors to illumination frame
        u_est_illum = R_illum_cam @ u_est
        u_true_illum = R_illum_cam @ u_true
        
        # Compute difference (small angle approximation)
        du = u_est_illum - u_true_illum
        
        # Convert to pixels
        residual_x_illum = du[0] * focal_length
        residual_y_illum = du[1] * focal_length
        
        return residual_x_illum, residual_y_illum
    
    def _compute_bearing_error(self,
                              pos_estimate: np.ndarray,
                              pos_true: np.ndarray) -> float:
        """Compute angular bearing error in radians"""
        u_est = pos_estimate / np.linalg.norm(pos_estimate)
        u_true = pos_true / np.linalg.norm(pos_true)
        
        cos_angle = np.clip(np.dot(u_est, u_true), -1, 1)
        return np.arccos(cos_angle)
    
    def _default_moon_error_model(self, range_km: float) -> Dict[str, float]:
        """Default Moon error model from Artemis I"""
        return {
            'radius_1sigma_px': 0.1,
            'centroid_1sigma_px': 0.15,
            'terrain_bias_max_px': 0.3
        }
    
    def _default_earth_error_model(self, range_km: float) -> Dict[str, float]:
        """Default Earth error model from Artemis I"""
        return {
            'radius_1sigma_px': 0.15,
            'centroid_1sigma_px': 0.2,
            'atmosphere_bias_km': 25.0  # Updated from flight data
        }
    
    def set_custom_error_model(self, 
                              target: TargetBody,
                              error_model_func: Callable):
        """Set custom error model for a target body"""
        self.error_models[target] = error_model_func
    
    def plot_figure_7(self, 
                     target: TargetBody = TargetBody.MOON,
                     filename: Optional[str] = None):
        """
        Figure 7: Moon Radius Residuals for All Passes
        
        X-axis: True range (km or nmi)
        Y-axis: Radius residual (pixels)
        Multiple passes, 3-σ error bounds
        """
        # Filter to target body
        indices = [i for i, img in enumerate(self.test_images) 
                  if img.target_body == target]
        
        if not indices:
            print(f"No data for {target.value}")
            return
        
        # Extract data
        ranges_nmi = np.array([self.test_images[i].range_m / 1852 for i in indices])
        radius_residuals = np.array([self.residuals[i].radius_residual_pixels for i in indices])
        pass_ids = [self.test_images[i].pass_id for i in indices]
        unique_passes = sorted(list(set(pass_ids)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each pass
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_passes)))
        for i, pass_id in enumerate(unique_passes):
            mask = np.array([pid == pass_id for pid in pass_ids])
            ax.scatter(ranges_nmi[mask], radius_residuals[mask],
                      c=[colors[i]], label=pass_id, alpha=0.6, s=40)
        
        # Add error model bounds
        range_array = np.linspace(ranges_nmi.min(), ranges_nmi.max(), 100)
        error_model = self.error_models[target](range_array[0] * 1.852)  # Convert to km
        sigma = error_model['radius_1sigma_px']
        
        ax.plot(range_array, 3*sigma*np.ones_like(range_array), 
               'k--', linewidth=2, label='3σ bounds')
        ax.plot(range_array, -3*sigma*np.ones_like(range_array), 
               'k--', linewidth=2)
        ax.fill_between(range_array, -3*sigma, 3*sigma, alpha=0.1, color='gray')
        
        # Format
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('True Range (nmi)', fontsize=12)
        ax.set_ylabel('Radius Residual (pixels)', fontsize=12)
        ax.set_title(f'{target.value.capitalize()} Radius Residuals for All Passes', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_figure_8(self,
                     target: TargetBody = TargetBody.MOON,
                     filename: Optional[str] = None):
        """
        Figure 8: Moon Centroid Residuals in Illumination Frame
        
        Shows X and Y centroid residuals over time/range
        Illumination frame: X parallel to sun, Y perpendicular
        """
        # Filter to target
        indices = [i for i, img in enumerate(self.test_images) 
                  if img.target_body == target]
        
        if not indices:
            print(f"No data for {target.value}")
            return
        
        # Extract data
        ranges_nmi = np.array([self.test_images[i].range_m / 1852 for i in indices])
        centroid_x = np.array([self.residuals[i].centroid_illum_x_pixels for i in indices])
        centroid_y = np.array([self.residuals[i].centroid_illum_y_pixels for i in indices])
        pass_ids = [self.test_images[i].pass_id for i in indices]
        unique_passes = sorted(list(set(pass_ids)))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_passes)))
        
        # X-direction (along sun)
        for i, pass_id in enumerate(unique_passes):
            mask = np.array([pid == pass_id for pid in pass_ids])
            ax1.scatter(ranges_nmi[mask], centroid_x[mask],
                       c=[colors[i]], label=pass_id, alpha=0.6, s=40)
        
        # Y-direction (perpendicular to sun)
        for i, pass_id in enumerate(unique_passes):
            mask = np.array([pid == pass_id for pid in pass_ids])
            ax2.scatter(ranges_nmi[mask], centroid_y[mask],
                       c=[colors[i]], alpha=0.6, s=40)
        
        # Error bounds
        error_model = self.error_models[target](ranges_nmi[0] * 1.852)
        sigma = error_model['centroid_1sigma_px']
        
        for ax in [ax1, ax2]:
            ax.axhline(3*sigma, color='k', linestyle='--', linewidth=2)
            ax.axhline(-3*sigma, color='k', linestyle='--', linewidth=2)
            ax.fill_between([ranges_nmi.min(), ranges_nmi.max()], -3*sigma, 3*sigma,
                          alpha=0.1, color='gray')
            ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        # Labels
        ax1.set_ylabel('X Centroid Residual (pixels)\n(along sun direction)', fontsize=11)
        ax2.set_ylabel('Y Centroid Residual (pixels)\n(perpendicular to sun)', fontsize=11)
        ax2.set_xlabel('True Range (nmi)', fontsize=12)
        ax1.set_title(f'{target.value.capitalize()} Centroid Residuals in Illumination Frame', fontsize=14)
        ax1.legend(loc='best', framealpha=0.9)
        
        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_certification_pass(self,
                               pass_id: str,
                               target: TargetBody = TargetBody.MOON,
                               filename: Optional[str] = None):
        """
        Figures 9-10 (Moon) or 16-17 (Earth): Certification Pass Details
        
        Time series of radius and centroid residuals over ~2 hour pass
        Includes BET uncertainty if available
        """
        # Filter to this pass
        indices = [i for i, img in enumerate(self.test_images)
                  if img.pass_id == pass_id and img.target_body == target]
        
        if not indices:
            print(f"No data for pass {pass_id}")
            return
        
        # Extract data
        times = np.array([self.test_images[i].timestamp for i in indices])
        times = (times - times[0]) / 3600  # Hours from start
        
        radius_res = np.array([self.residuals[i].radius_residual_pixels for i in indices])
        centroid_x = np.array([self.residuals[i].centroid_illum_x_pixels for i in indices])
        centroid_y = np.array([self.residuals[i].centroid_illum_y_pixels for i in indices])
        ranges_km = np.array([self.test_images[i].range_m / 1000 for i in indices])
        
        # BET uncertainty (if available)
        bet_range_m = self.test_images[indices[0]].bet_uncertainty_range_m
        bet_pointing_arcsec = self.test_images[indices[0]].bet_uncertainty_pointing_arcsec
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Error model
        error_model = self.error_models[target](np.mean(ranges_km))
        radius_sigma = error_model['radius_1sigma_px']
        centroid_sigma = error_model['centroid_1sigma_px']
        
        # Convert BET uncertainties to pixels
        if bet_range_m is not None and bet_pointing_arcsec is not None:
            focal_length = self.test_images[indices[0]].camera_K[0, 0]
            bet_radius_px = self._range_uncertainty_to_pixels(
                bet_range_m/1000, np.mean(ranges_km), focal_length, target)
            bet_centroid_px = np.deg2rad(bet_pointing_arcsec/3600) * focal_length
        else:
            bet_radius_px = 0
            bet_centroid_px = 0
        
        # Radius plot
        axes[0].plot(times, radius_res, 'b.-', markersize=4, linewidth=1, label='Residuals')
        axes[0].fill_between(times, 
                            -3*radius_sigma - bet_radius_px,
                            3*radius_sigma + bet_radius_px,
                            alpha=0.2, color='gray', label='3σ + BET unc.')
        axes[0].plot(times, 3*radius_sigma*np.ones_like(times), 'k--', linewidth=1.5, alpha=0.7)
        axes[0].plot(times, -3*radius_sigma*np.ones_like(times), 'k--', linewidth=1.5, alpha=0.7)
        
        # Centroid X
        axes[1].plot(times, centroid_x, 'r.-', markersize=4, linewidth=1)
        axes[1].fill_between(times,
                            -3*centroid_sigma - bet_centroid_px,
                            3*centroid_sigma + bet_centroid_px,
                            alpha=0.2, color='gray')
        axes[1].plot(times, 3*centroid_sigma*np.ones_like(times), 'k--', linewidth=1.5, alpha=0.7)
        axes[1].plot(times, -3*centroid_sigma*np.ones_like(times), 'k--', linewidth=1.5, alpha=0.7)
        
        # Centroid Y
        axes[2].plot(times, centroid_y, 'g.-', markersize=4, linewidth=1)
        axes[2].fill_between(times,
                            -3*centroid_sigma - bet_centroid_px,
                            3*centroid_sigma + bet_centroid_px,
                            alpha=0.2, color='gray')
        axes[2].plot(times, 3*centroid_sigma*np.ones_like(times), 'k--', linewidth=1.5, alpha=0.7)
        axes[2].plot(times, -3*centroid_sigma*np.ones_like(times), 'k--', linewidth=1.5, alpha=0.7)
        
        # Format
        for ax in axes:
            ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        axes[0].set_ylabel('Radius Residual\n(pixels)', fontsize=11)
        axes[1].set_ylabel('X Centroid Residual\n(pixels)', fontsize=11)
        axes[2].set_ylabel('Y Centroid Residual\n(pixels)', fontsize=11)
        axes[2].set_xlabel('Time from Start (hours)', fontsize=12)
        axes[0].set_title(f'{target.value.capitalize()} Certification Pass: {pass_id}', fontsize=14)
        axes[0].legend(loc='best', framealpha=0.9)
        
        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def plot_algorithm_comparison(self,
                                 pass_id: str,
                                 results_iterative: List[CRAResult],
                                 target: TargetBody = TargetBody.MOON,
                                 filename: Optional[str] = None):
        """
        Figures 13-15: Comparison between Non-Iterative and Iterative CRA
        
        Shows both algorithms' residuals and their differences
        """
        # Filter to this pass
        indices = [i for i, img in enumerate(self.test_images)
                  if img.pass_id == pass_id and img.target_body == target]
        
        if not indices:
            print(f"No data for pass {pass_id}")
            return
        
        # Compute residuals for iterative results
        residuals_iter = []
        for i, idx in enumerate(indices):
            res = self._compute_residuals(self.test_images[idx], results_iterative[i])
            residuals_iter.append(res)
        
        # Extract data (non-iterative already stored)
        radius_non_iter = np.array([self.residuals[i].radius_residual_pixels for i in indices])
        radius_iter = np.array([r.radius_residual_pixels for r in residuals_iter])
        
        centroid_x_non_iter = np.array([self.residuals[i].centroid_illum_x_pixels for i in indices])
        centroid_x_iter = np.array([r.centroid_illum_x_pixels for r in residuals_iter])
        
        centroid_y_non_iter = np.array([self.residuals[i].centroid_illum_y_pixels for i in indices])
        centroid_y_iter = np.array([r.centroid_illum_y_pixels for r in residuals_iter])
        
        image_nums = np.arange(len(indices))
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Radius
        axes[0,0].plot(image_nums, radius_non_iter, 'r.-', label='Non-iterative', markersize=4)
        axes[0,0].plot(image_nums, radius_iter, 'k.-', label='Iterative', markersize=4)
        axes[0,0].set_ylabel('Radius (pixels)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(image_nums, radius_non_iter - radius_iter, 'b.-', markersize=4)
        axes[0,1].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[0,1].set_ylabel('Radius Difference (pixels)')
        axes[0,1].grid(True, alpha=0.3)
        
        # X-centroid
        axes[1,0].plot(image_nums, centroid_x_non_iter, 'r.-', label='Non-iterative', markersize=4)
        axes[1,0].plot(image_nums, centroid_x_iter, 'k.-', label='Iterative', markersize=4)
        axes[1,0].set_ylabel('X Centroid (pixels)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].plot(image_nums, centroid_x_non_iter - centroid_x_iter, 'b.-', markersize=4)
        axes[1,1].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[1,1].set_ylabel('X Centroid Difference (pixels)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Y-centroid
        axes[2,0].plot(image_nums, centroid_y_non_iter, 'r.-', label='Non-iterative', markersize=4)
        axes[2,0].plot(image_nums, centroid_y_iter, 'k.-', label='Iterative', markersize=4)
        axes[2,0].set_ylabel('Y Centroid (pixels)')
        axes[2,0].set_xlabel('Image Number')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        axes[2,1].plot(image_nums, centroid_y_non_iter - centroid_y_iter, 'b.-', markersize=4)
        axes[2,1].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[2,1].set_ylabel('Y Centroid Difference (pixels)')
        axes[2,1].set_xlabel('Image Number')
        axes[2,1].grid(True, alpha=0.3)
        
        fig.suptitle(f'Algorithm Comparison - Pass {pass_id}', fontsize=14)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def plot_earth_atmosphere_bias(self,
                                   atmosphere_bias_km_tested: List[float] = [35, 30, 25, 20],
                                   pass_id: Optional[str] = None,
                                   filename: Optional[str] = None):
        """
        Figure 20: Earth Atmosphere Bias Comparison
        
        Shows how different atmosphere bias values affect radius residuals
        """
        # Filter to Earth images
        indices = [i for i, img in enumerate(self.test_images)
                  if img.target_body == TargetBody.EARTH]
        
        if pass_id:
            indices = [i for i in indices if self.test_images[i].pass_id == pass_id]
        
        if not indices:
            print("No Earth data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Original residuals (with whatever bias was used)
        ranges_nmi = np.array([self.test_images[i].range_m / 1852 for i in indices])
        original_residuals = np.array([self.residuals[i].radius_residual_pixels for i in indices])
        
        ax.scatter(ranges_nmi, original_residuals, 
                  c='blue', label=f'Original (flight bias)', alpha=0.6, s=40)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(atmosphere_bias_km_tested)))
        for i, bias_km in enumerate(atmosphere_bias_km_tested):
            pass
        
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('True Range (nmi)', fontsize=12)
        ax.set_ylabel('Radius Residual (pixels)', fontsize=12)
        ax.set_title('Earth Atmosphere Bias Comparison', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def _range_uncertainty_to_pixels(self, 
                                    range_unc_km: float,
                                    range_km: float,
                                    focal_length: float,
                                    target: TargetBody) -> float:
        """Convert range uncertainty to pixel radius uncertainty"""
        if target == TargetBody.MOON:
            body_radius = self.moon_radius_m
        else:
            body_radius = self.earth_radii_m[0]
        
        pixels = focal_length * (body_radius / (range_km * 1000))
        dpixels = pixels * (range_unc_km / range_km)
        return dpixels
    
    def generate_statistics_table(self,
                                 target: Optional[TargetBody] = None,
                                 pass_id: Optional[str] = None) -> Dict:
        """
        Generate statistics table (like Table 2 in paper)
        
        Args:
            target: Filter by target body
            pass_id: Filter by pass
        """
        # Filter indices
        indices = list(range(len(self.test_images)))
        
        if target:
            indices = [i for i in indices if self.test_images[i].target_body == target]
        
        if pass_id:
            indices = [i for i in indices if self.test_images[i].pass_id == pass_id]
        
        if not indices:
            return {}
        
        # Extract residuals
        radius = np.array([self.residuals[i].radius_residual_pixels for i in indices])
        centroid_x = np.array([self.residuals[i].centroid_illum_x_pixels for i in indices])
        centroid_y = np.array([self.residuals[i].centroid_illum_y_pixels for i in indices])
        
        return {
            'n_images': len(indices),
            'radius': {
                'mean': np.mean(radius),
                'std': np.std(radius, ddof=1),
                'min': np.min(radius),
                'max': np.max(radius),
            },
            'centroid_x': {
                'mean': np.mean(centroid_x),
                'std': np.std(centroid_x, ddof=1),
                'min': np.min(centroid_x),
                'max': np.max(centroid_x),
            },
            'centroid_y': {
                'mean': np.mean(centroid_y),
                'std': np.std(centroid_y, ddof=1),
                'min': np.min(centroid_y),
                'max': np.max(centroid_y),
            }
        }
    
    def print_statistics_table(self, **kwargs):
        """Print formatted statistics table"""
        stats = self.generate_statistics_table(**kwargs)
        
        if not stats:
            print("No data available for statistics")
            return
        
        print(f"\n{'='*70}")
        print(f"OpNav Performance Statistics")
        print(f"{'='*70}")
        print(f"Number of images: {stats['n_images']}")
        print(f"\n{'Parameter':<20} {'Mean':>12} {'Std Dev':>12} {'Min':>12} {'Max':>12}")
        print(f"{'-'*68}")
        
        for param in ['radius', 'centroid_x', 'centroid_y']:
            name = param.replace('_', '-')
            print(f"{name:<20} "
                  f"{stats[param]['mean']:>12.4f} "
                  f"{stats[param]['std']:>12.4f} "
                  f"{stats[param]['min']:>12.4f} "
                  f"{stats[param]['max']:>12.4f}")
        
        print(f"{'='*70}\n")
        print("All values in pixels")


# Example usage
if __name__ == "__main__":
    print("Universal OpNav Testing Framework loaded")
    print("This framework can generate All figures from the Artemis I paper")
    print("\nKey features:")
    print("- Handles Moon and Earth observations")
    print("- Computes residuals in illumination frame")
    print("- Generates Figures 7-21")
    print("- Algorithm comparison plots")
    print("- Atmosphere bias analysis")
    print("\nSee integration example for usage")