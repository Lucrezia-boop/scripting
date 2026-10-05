# -*- coding: utf-8 -*-
"""
Universal Orienter Script for Rhino
====================================
Orients any combination of geometry so user-defined ground-contact points
lie on a destination plane. The destination is either:

  * World XY (default), or
  * the local frame of a selected flat surface, evaluated at the surface's
    area centroid (the "geometry-dictated local XY" of that surface).

Workflow:
1. Pre-select objects to orient (ENTER to continue).
2. Optionally select a surface to orient onto, or ENTER for World XY.
3. Place 3+ points where ground contact should be, ENTER to confirm.
4. Script creates a copy that preserves ALL object attributes
   (block definitions, layer, display colour, print colour, linetype,
   material, name).
   - World XY mode: the copy is offset on +Y by 2x the largest world X/Y
     bounding-box extent of the selection (in model units).
   - Surface mode: the copy is placed on the surface itself.
5. Rotates/levels the copy so the fitted ground plane aligns with the
   destination plane and the lowest reference point rests on it.
6. Surface mode only: shows a light preview with a clickable "Flip" option
   (F) to send the object to the opposite side of the surface, since the
   surface frame may point inwards when the user wants it outwards.
7. Cleans up reference points.

ESC cancels at any point, cleaning up any temporary geometry.

Author: Claude Code Assistant
Location: rhino-py/UniversalOrienter.py
"""

import math
import rhinoscriptsyntax as rs
import scriptcontext as sc
import Rhino
from Rhino.Geometry import Point3d, Vector3d, Plane, Transform, BoundingBox


# =============================================================================
# CONFIGURATION
# =============================================================================
OFFSET_FACTOR = 2.0    # World XY mode: planar offset = largest X/Y extent x this
MIN_POINTS = 3         # Minimum ground contact points required
COLLINEAR_TOL = 0.001  # Tolerance for collinearity detection
MAX_LEVEL_ITERS = 10   # Iterations for non-coplanar leveling


def _unit_name():
    """Abbreviated model unit name (mm, cm, m, ...) for user messages."""
    try:
        return rs.UnitSystemName(False, True, True)
    except Exception:
        return "units"


def _level_tolerance():
    """Leveling tolerance in model units, derived from the document tolerance
    so the script behaves correctly in mm, cm or m."""
    t = sc.doc.ModelAbsoluteTolerance
    return t if t and t > 0 else 1e-6


# =============================================================================
# CLEANUP MANAGER - Ensures cleanup on cancel/error
# =============================================================================
class CleanupManager:
    """
    Tracks temporary geometry for safe cleanup on cancel or error.
    Use with 'with' statement for automatic cleanup.
    """
    def __init__(self):
        self.temp_point_ids = []
        self.copied_object_ids = []
        self.cancelled = False

    def add_temp_point(self, pt_id):
        if pt_id:
            self.temp_point_ids.append(pt_id)

    def add_copied_object(self, obj_id):
        if obj_id:
            self.copied_object_ids.append(obj_id)

    def mark_cancelled(self):
        self.cancelled = True

    def cleanup_on_cancel(self):
        """Delete all tracked geometry - used when operation is cancelled."""
        rs.EnableRedraw(False)
        # Delete temp points
        for pt_id in self.temp_point_ids:
            if pt_id and rs.IsObject(pt_id):
                rs.DeleteObject(pt_id)
        # Delete copied objects (only on cancel)
        for obj_id in self.copied_object_ids:
            if obj_id and rs.IsObject(obj_id):
                rs.DeleteObject(obj_id)
        rs.EnableRedraw(True)
        self.temp_point_ids = []
        self.copied_object_ids = []

    def cleanup_points_only(self):
        """Delete only reference points - used on successful completion."""
        rs.EnableRedraw(False)
        for pt_id in self.temp_point_ids:
            if pt_id and rs.IsObject(pt_id):
                rs.DeleteObject(pt_id)
        rs.EnableRedraw(True)
        self.temp_point_ids = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cancelled or exc_type is not None:
            self.cleanup_on_cancel()
            if exc_type is not None:
                print("Error occurred: {}".format(exc_val))
        return False  # Don't suppress exceptions


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def get_combined_bounding_box(object_ids):
    """Get combined, world-axis-aligned bounding box for multiple objects."""
    bbox = BoundingBox.Empty
    for obj_id in object_ids:
        obj = rs.coercegeometry(obj_id)
        if obj:
            obj_bbox = obj.GetBoundingBox(True)
            bbox.Union(obj_bbox)
        else:
            # Fallback for objects that don't coerce (V-Ray, etc.)
            corners = rs.BoundingBox(obj_id)
            if corners:
                for corner in corners:
                    bbox.Union(Point3d(corner[0], corner[1], corner[2]))
    return bbox


def check_collinearity(points):
    """
    Check if points are collinear (cannot define a plane).
    Returns True if collinear (bad), False if OK.
    """
    if len(points) < 3:
        return True

    # Use first 3 points
    p0 = points[0] if isinstance(points[0], Point3d) else Point3d(points[0][0], points[0][1], points[0][2])
    p1 = points[1] if isinstance(points[1], Point3d) else Point3d(points[1][0], points[1][1], points[1][2])
    p2 = points[2] if isinstance(points[2], Point3d) else Point3d(points[2][0], points[2][1], points[2][2])

    v1 = p1 - p0
    v2 = p2 - p0
    cross = Vector3d.CrossProduct(v1, v2)

    return cross.Length < COLLINEAR_TOL


def fit_plane_to_points(points):
    """
    Fit a plane to a list of points.
    Returns (success, plane) tuple.
    """
    # Convert to Point3d list if needed
    pt3d_list = []
    for pt in points:
        if isinstance(pt, Point3d):
            pt3d_list.append(pt)
        else:
            pt3d_list.append(Point3d(pt[0], pt[1], pt[2]))

    # Use RhinoCommon's FitPlaneToPoints
    result, plane = Plane.FitPlaneToPoints(pt3d_list)

    # result is PlaneFitResult enum: Success, Failure, etc.
    if result == Rhino.Geometry.PlaneFitResult.Success:
        return True, plane
    elif result == Rhino.Geometry.PlaneFitResult.Failure:
        return False, None
    else:
        # Might be "Inconclusive" - try anyway if plane is valid
        if plane and plane.IsValid:
            return True, plane
        return False, None


def ensure_normal_points_up(plane, reference_centroid):
    """
    Ensure the plane normal points towards the object body (reference centroid).
    This defines which face of the object becomes the contact face.
    """
    to_centroid = reference_centroid - plane.Origin
    dot = Vector3d.Multiply(plane.Normal, to_centroid)
    if dot < 0:
        plane.Flip()
    return plane


def get_points_centroid(points):
    """Calculate centroid of a list of points (Point3d or coordinate triples)."""
    sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
    for pt in points:
        if isinstance(pt, Point3d):
            sum_x += pt.X
            sum_y += pt.Y
            sum_z += pt.Z
        else:
            sum_x += pt[0]
            sum_y += pt[1]
            sum_z += pt[2]
    n = len(points)
    return Point3d(sum_x / n, sum_y / n, sum_z / n)


def point_ids_to_point3d(point_ids):
    """Read current coordinates of point objects as Point3d list."""
    pts = []
    for pid in point_ids:
        c = rs.PointCoordinates(pid)
        if c:
            pts.append(Point3d(c[0], c[1], c[2]))
    return pts


def plane_relative_heights(dest_plane, point_ids):
    """Signed distance of each reference point from dest_plane (+ = normal side)."""
    return [dest_plane.DistanceTo(p) for p in point_ids_to_point3d(point_ids)]


# =============================================================================
# DESTINATION PLANE FROM A SELECTED SURFACE
# =============================================================================

def _surface_and_centroid_from_objref(objref):
    """
    From a GetObject result, return (surface, area_centroid), honoring a
    picked sub-face when present. 'surface' is a Surface/BrepFace.
    """
    brep = objref.Brep()
    ci = objref.GeometryComponentIndex
    face = None
    if brep is not None:
        if (ci.ComponentIndexType == Rhino.Geometry.ComponentIndexType.BrepFace
                and 0 <= ci.Index < brep.Faces.Count):
            face = brep.Faces[ci.Index]
        elif brep.Faces.Count > 0:
            face = brep.Faces[0]

    if face is not None:
        single = face.DuplicateFace(False)  # Brep of just this face
        amp = Rhino.Geometry.AreaMassProperties.Compute(single)
        if amp:
            centroid = amp.Centroid
        else:
            centroid = face.PointAt(face.Domain(0).Mid, face.Domain(1).Mid)
        return face, centroid

    srf = objref.Surface()
    if srf is not None:
        amp = Rhino.Geometry.AreaMassProperties.Compute(srf)
        if amp:
            centroid = amp.Centroid
        else:
            centroid = srf.PointAt(srf.Domain(0).Mid, srf.Domain(1).Mid)
        return srf, centroid

    return None, None


def destination_plane_from_objref(objref):
    """
    Build the destination plane from a surface: the surface's local frame
    evaluated at its area centroid. Returns a Plane or None.
    """
    surface, centroid = _surface_and_centroid_from_objref(objref)
    if surface is None or centroid is None:
        return None

    ok, u, v = surface.ClosestPoint(centroid)
    if not ok:
        return None

    ok2, frame = surface.FrameAt(u, v)
    if not ok2 or frame is None or not frame.IsValid:
        return None

    # Prefer the face's outward orientation as the default "up" side.
    if getattr(surface, "OrientationIsReversed", False):
        frame = Plane(frame.Origin, frame.XAxis, -frame.YAxis)

    return frame


def select_orientation_surface():
    """
    Optional surface selection, presented before ground-point placement.

    Returns:
        ('surface', dest_plane) if a flat surface was chosen,
        ('xy', None)            to orient to World XY (user pressed ENTER),
        ('cancel', None)        if the user pressed ESC.
    """
    go = Rhino.Input.Custom.GetObject()
    go.SetCommandPrompt(
        "Select a flat surface to orient onto, or press Enter to orient to World XY")
    go.GeometryFilter = (Rhino.DocObjects.ObjectType.Surface |
                         Rhino.DocObjects.ObjectType.Brep)
    go.SubObjectSelect = True
    go.EnablePreSelect(False, False)       # objects were already chosen
    go.DeselectAllBeforePostSelect = False
    go.AcceptNothing(True)

    res = go.Get()

    if res == Rhino.Input.GetResult.Cancel:
        return 'cancel', None
    if res == Rhino.Input.GetResult.Nothing:
        return 'xy', None
    if res == Rhino.Input.GetResult.Object and go.ObjectCount > 0:
        dest = destination_plane_from_objref(go.Object(0))
        if dest is None:
            print("Could not evaluate that surface - orienting to World XY instead.")
            return 'xy', None
        print("Orienting onto selected surface (frame at its area centroid).")
        return 'surface', dest

    return 'cancel', None


# =============================================================================
# LEVELING (relative to an arbitrary destination plane)
# =============================================================================

def iterative_leveling(object_ids, point_ids, dest_plane,
                       max_iterations=MAX_LEVEL_ITERS, tolerance=None):
    """
    Iteratively apply small rotations to minimize the spread of the reference
    points measured perpendicular to dest_plane. Handles non-coplanar points
    by finding a best-compromise rotation.

    Returns (final_spread, iterations_performed).
    """
    if tolerance is None:
        tolerance = _level_tolerance()

    normal = dest_plane.Normal

    for iteration in range(max_iterations):
        pts = point_ids_to_point3d(point_ids)
        if len(pts) < 3:
            break

        heights = [dest_plane.DistanceTo(p) for p in pts]
        min_h = min(heights)
        max_h = max(heights)
        spread = max_h - min_h

        if spread <= tolerance:
            return spread, iteration

        centroid = get_points_centroid(pts)
        low = pts[heights.index(min_h)]
        high = pts[heights.index(max_h)]

        # Work in destination-plane space: X,Y in-plane, Z = signed height.
        ok_l, lp = dest_plane.RemapToPlaneSpace(low)
        ok_h, hp = dest_plane.RemapToPlaneSpace(high)
        if not (ok_l and ok_h):
            break

        dx = hp.X - lp.X
        dy = hp.Y - lp.Y
        horizontal_dist = math.sqrt(dx * dx + dy * dy)
        if horizontal_dist < 1e-9:
            # Points stacked along the normal - rotation cannot help.
            break

        dz = hp.Z - lp.Z  # == max_h - min_h
        # Half the angle for gradual, overshoot-free correction.
        angle_rad = math.atan2(dz, horizontal_dist) * 0.5

        # Rotation axis: perpendicular to the high-low vector, lying in the
        # plane. In plane space that is (-dy, dx, 0); map back to world.
        axis = dest_plane.XAxis * (-dy) + dest_plane.YAxis * dx
        if not axis.Unitize():
            break

        rotation_xform = Transform.Rotation(angle_rad, axis, centroid)

        rs.EnableRedraw(False)
        for obj_id in object_ids:
            rs.TransformObject(obj_id, rotation_xform, copy=False)
        for pt_id in point_ids:
            rs.TransformObject(pt_id, rotation_xform, copy=False)
        rs.EnableRedraw(True)

    final = plane_relative_heights(dest_plane, point_ids)
    final_spread = (max(final) - min(final)) if final else 0.0
    return final_spread, max_iterations


def level_onto_plane(object_ids, point_ids, dest_plane):
    """
    Translate objects + points along the destination normal so the lowest
    reference point rests exactly on dest_plane. Returns the shift applied.
    """
    heights = plane_relative_heights(dest_plane, point_ids)
    if not heights:
        return 0.0

    min_h = min(heights)
    move = dest_plane.Normal * (-min_h)
    xform = Transform.Translation(move)

    rs.EnableRedraw(False)
    for obj_id in object_ids:
        rs.TransformObject(obj_id, xform, copy=False)
    for pt_id in point_ids:
        rs.TransformObject(pt_id, xform, copy=False)
    rs.EnableRedraw(True)
    return min_h


def report_point_deviations(point_ids, dest_plane):
    """
    Report each reference point's signed distance from the destination plane.
    Returns (report_string, max_abs_deviation).
    """
    heights = plane_relative_heights(dest_plane, point_ids)
    if not heights:
        return "No points to report.", 0

    unit = _unit_name()
    max_dev = max(abs(h) for h in heights)

    if max_dev < _level_tolerance() * 10:
        return "All reference points lie on the destination plane.", 0

    lines = ["Reference point deviations from destination plane:"]
    for i, h in enumerate(heights):
        lines.append("  Point {}: {:.3f} {}".format(i + 1, h, unit))
    lines.append("Max deviation: {:.3f} {}".format(max_dev, unit))
    return "\n".join(lines), max_dev


# =============================================================================
# FLIP PREVIEW (surface mode only)
# =============================================================================

def _apply_flip(object_ids, point_ids, dest_plane):
    """
    Send the oriented objects to the opposite side of the surface with a rigid
    180-degree rotation about the destination plane's X axis (NOT a mirror, so
    geometry and handedness are preserved), then re-level onto the plane.
    """
    pts = point_ids_to_point3d(point_ids)
    pivot = get_points_centroid(pts) if pts else dest_plane.Origin
    xform = Transform.Rotation(math.pi, dest_plane.XAxis, pivot)

    rs.EnableRedraw(False)
    for obj_id in object_ids:
        rs.TransformObject(obj_id, xform, copy=False)
    for pt_id in point_ids:
        rs.TransformObject(pt_id, xform, copy=False)
    rs.EnableRedraw(True)

    level_onto_plane(object_ids, point_ids, dest_plane)


def flip_preview(object_ids, point_ids, dest_plane, cleanup_mgr):
    """
    Light preview loop with a clickable 'Flip' option (F underlined on the
    command line). The oriented copy is already in the document, so the preview
    is the real result. Returns True if accepted (ENTER), False on ESC.
    """
    sc.doc.Views.Redraw()

    while True:
        go = Rhino.Input.Custom.GetOption()
        go.SetCommandPrompt(
            "Orientation preview: press Enter to accept, "
            "or Flip to the opposite side of the surface")
        flip_idx = go.AddOption("Flip")
        go.AcceptNothing(True)

        res = go.Get()

        if res == Rhino.Input.GetResult.Cancel:
            cleanup_mgr.mark_cancelled()
            return False
        if res == Rhino.Input.GetResult.Nothing:
            return True
        if res == Rhino.Input.GetResult.Option and go.OptionIndex() == flip_idx:
            _apply_flip(object_ids, point_ids, dest_plane)
            sc.doc.Views.Redraw()
            continue
        # Any other result: keep previewing.


# =============================================================================
# INTERACTIVE POINT COLLECTION (with proper ESC handling)
# =============================================================================

def collect_ground_points(cleanup_mgr):
    """
    Interactively collect ground contact points from user.
    Returns list of point IDs, or None if cancelled.
    """
    point_ids = []

    while True:
        count = len(point_ids)
        if count < MIN_POINTS:
            remaining = MIN_POINTS - count
            prompt = "Place ground contact point {} ({} more required). ESC to cancel".format(
                count + 1, remaining)
        else:
            prompt = "Place point {} or press ENTER to finish ({} placed). ESC to cancel".format(
                count + 1, count)

        gp = Rhino.Input.Custom.GetPoint()
        gp.SetCommandPrompt(prompt)

        if count >= MIN_POINTS:
            gp.AcceptNothing(True)

        result = gp.Get()

        if result == Rhino.Input.GetResult.Cancel:
            print("Cancelled by user.")
            cleanup_mgr.mark_cancelled()
            return None

        elif result == Rhino.Input.GetResult.Nothing:
            if count >= MIN_POINTS:
                coords = [rs.PointCoordinates(pid) for pid in point_ids]
                if check_collinearity(coords):
                    print("Points are collinear - cannot define a plane. Add a non-collinear point.")
                    continue
                else:
                    print("Accepted {} ground contact points.".format(count))
                    return point_ids
            else:
                print("Need at least {} points. Keep placing...".format(MIN_POINTS))
                continue

        elif result == Rhino.Input.GetResult.Point:
            pt = gp.Point()
            pt_id = rs.AddPoint(pt)
            if pt_id:
                point_ids.append(pt_id)
                cleanup_mgr.add_temp_point(pt_id)

        else:
            print("Unexpected input. Operation cancelled.")
            cleanup_mgr.mark_cancelled()
            return None

    return point_ids


# =============================================================================
# OBJECT COPYING / TRANSFORMING (attribute preservation)
# =============================================================================

def remove_from_all_groups(obj_id):
    """Remove an object from all groups, breaking the link to the originals."""
    groups = rs.ObjectGroups(obj_id)
    if groups:
        for group_name in groups:
            rs.RemoveObjectFromGroup(obj_id, group_name)


def _find_rhino_object(obj_id):
    """Resolve a GUID to a RhinoObject across Rhino versions."""
    try:
        obj = sc.doc.Objects.FindId(obj_id)
        if obj:
            return obj
    except Exception:
        pass
    try:
        return sc.doc.Objects.Find(obj_id)
    except Exception:
        return None


def restamp_attributes(source_id, target_id):
    """
    Safeguard that forces a copy to carry the EXACT attributes of its source:
    layer, display (object) colour, print (plot) colour, plot weight, linetype,
    material and name. rs.TransformObject(copy=True) already duplicates these
    via doc.Objects.Transform, and block instances keep their definition; this
    guarantees preservation even on the command-line fallback path.
    """
    try:
        src = _find_rhino_object(source_id)
        tgt = _find_rhino_object(target_id)
        if not src or not tgt:
            return

        sa = src.Attributes
        ta = tgt.Attributes

        ta.LayerIndex = sa.LayerIndex
        ta.ColorSource = sa.ColorSource
        ta.ObjectColor = sa.ObjectColor
        ta.PlotColorSource = sa.PlotColorSource
        ta.PlotColor = sa.PlotColor
        ta.PlotWeightSource = sa.PlotWeightSource
        ta.PlotWeight = sa.PlotWeight
        ta.LinetypeSource = sa.LinetypeSource
        ta.LinetypeIndex = sa.LinetypeIndex
        ta.MaterialSource = sa.MaterialSource
        ta.MaterialIndex = sa.MaterialIndex
        if sa.Name:
            ta.Name = sa.Name

        sc.doc.Objects.ModifyAttributes(target_id, ta, True)
    except Exception as e:
        print("Warning: could not re-stamp attributes for {}: {}".format(target_id, e))


def transform_objects(object_ids, xform, cleanup_mgr, copy=False):
    """
    Transform objects with handling for all object types.
    Returns list of (possibly new) object IDs.

    When copying, removes group associations from copies (to prevent
    cross-selection) but preserves every other attribute.
    """
    result_ids = []

    for obj_id in object_ids:
        try:
            new_id = rs.TransformObject(obj_id, xform, copy=copy)
            if new_id:
                if copy:
                    remove_from_all_groups(new_id)
                    restamp_attributes(obj_id, new_id)
                    cleanup_mgr.add_copied_object(new_id)
                result_ids.append(new_id)
            else:
                # TransformObject failed - command-line fallback (V-Ray, etc.)
                print("Warning: Transform failed for object {}. Trying fallback...".format(obj_id))
                if copy:
                    rs.SelectObject(obj_id)
                    rs.Command("_Copy 0,0,0 0,0,0 _Enter", False)
                    new_objs = rs.LastCreatedObjects()
                    if new_objs:
                        new_id = new_objs[0]
                        remove_from_all_groups(new_id)
                        rs.TransformObject(new_id, xform, copy=False)
                        restamp_attributes(obj_id, new_id)
                        cleanup_mgr.add_copied_object(new_id)
                        result_ids.append(new_id)
                    rs.UnselectAllObjects()
                else:
                    result_ids.append(obj_id)
        except Exception as e:
            print("Error transforming object {}: {}".format(obj_id, e))
            result_ids.append(obj_id)

    return result_ids


def compute_planar_offset(object_ids):
    """
    World XY mode offset: 2x the largest world X/Y bounding-box extent of the
    selection, as a +Y vector. Bounding box is world-axis-aligned (not local),
    and extents are already in model units, so this works for mm/cm/m directly.
    Returns (offset_vector, distance).
    """
    bbox = get_combined_bounding_box(object_ids)
    dx = bbox.Max.X - bbox.Min.X
    dy = bbox.Max.Y - bbox.Min.Y
    dz = bbox.Max.Z - bbox.Min.Z

    dist = max(dx, dy) * OFFSET_FACTOR
    if dist <= _level_tolerance():
        # Degenerate in XY (e.g. a vertical object): fall back to any extent.
        dist = max(dx, dy, dz) * OFFSET_FACTOR
    if dist <= _level_tolerance():
        dist = 1.0

    return Vector3d(0, dist, 0), dist


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def universal_orient():
    """Main function for Universal Orienter."""
    print("=" * 50)
    print("UNIVERSAL ORIENTER")
    print("=" * 50)

    unit = _unit_name()

    # --- Step 1: Get objects (pre-selection allowed) ---
    object_ids = rs.GetObjects(
        "Objects are pre-selected. Press ENTER to continue, or select more objects.",
        preselect=True
    )

    if not object_ids:
        print("No objects selected. Operation cancelled.")
        return

    print("Selected {} objects.".format(len(object_ids)))
    rs.UnselectAllObjects()

    # --- Step 2: Optional destination surface ---
    mode, dest_plane = select_orientation_surface()
    if mode == 'cancel':
        print("Cancelled by user.")
        return

    # --- Use CleanupManager for safe cleanup ---
    with CleanupManager() as cleanup:

        # --- Step 3: Collect ground contact points ---
        print("\nPlace points where ground contact should be (minimum {}).".format(MIN_POINTS))
        print("Press ENTER when done, or ESC to cancel.")

        point_ids = collect_ground_points(cleanup)
        if point_ids is None:
            return

        original_point_count = len(point_ids)

        # --- Step 4: Create attribute-preserving copies ---
        # Always offset the working copy by a real (non-identity) distance so
        # doc.Objects.Transform reliably duplicates. In surface mode the offset
        # is irrelevant to the result, since PlaneToPlane (Step 5) relocates the
        # copy onto the surface regardless of where it started.
        offset_vector, offset_dist = compute_planar_offset(object_ids)
        if mode == 'xy':
            print("\nCreating offset copy ({:.3f} {} on +Y)...".format(offset_dist, unit))
        else:
            print("\nCreating working copy to place on the selected surface...")

        offset_xform = Transform.Translation(offset_vector)

        rs.EnableRedraw(False)

        copied_ids = transform_objects(object_ids, offset_xform, cleanup, copy=True)
        if not copied_ids:
            print("Failed to create copies. Operation cancelled.")
            cleanup.mark_cancelled()
            rs.EnableRedraw(True)
            return

        copied_point_ids = []
        for pt_id in point_ids:
            new_pt_id = rs.TransformObject(pt_id, offset_xform, copy=True)
            if new_pt_id:
                copied_point_ids.append(new_pt_id)
                cleanup.add_temp_point(new_pt_id)

        rs.EnableRedraw(True)
        print("Created {} copied objects.".format(len(copied_ids)))

        copied_points = point_ids_to_point3d(copied_point_ids)

        # --- Step 5: Fit plane and build the orientation transform ---
        print("\nCalculating orientation...")

        success, source_plane = fit_plane_to_points(copied_points)
        if not success:
            print("Failed to fit plane to points. Points may be collinear or too close.")
            cleanup.mark_cancelled()
            return

        # Normal should point toward the object body (defines the contact face).
        copied_centroid = get_combined_bounding_box(copied_ids).Center
        source_plane = ensure_normal_points_up(source_plane, copied_centroid)

        # Pin the source origin to the contact centroid for an exact mapping.
        ref_centroid = get_points_centroid(copied_points)
        source_plane.Origin = ref_centroid

        # Destination plane: World XY at the contact centroid, or surface frame.
        if mode == 'xy':
            target_plane = Plane(ref_centroid, Vector3d.ZAxis)
        else:
            target_plane = Plane(dest_plane)  # copy; origin on the surface

        rotation_xform = Transform.PlaneToPlane(source_plane, target_plane)

        # --- Step 6: Apply rotation ---
        print("Applying rotation...")
        rs.EnableRedraw(False)
        for obj_id in copied_ids:
            rs.TransformObject(obj_id, rotation_xform, copy=False)
        for pt_id in copied_point_ids:
            rs.TransformObject(pt_id, rotation_xform, copy=False)
        rs.EnableRedraw(True)

        # --- Step 7: Leveling for non-coplanar points ---
        tol = _level_tolerance()
        heights = plane_relative_heights(target_plane, copied_point_ids)
        initial_spread = (max(heights) - min(heights)) if heights else 0.0

        if initial_spread > tol:
            print("Initial spread: {:.3f} {} - applying iterative leveling...".format(
                initial_spread, unit))
            final_spread, iterations = iterative_leveling(
                copied_ids, copied_point_ids, target_plane, tolerance=tol)
            print("After {} iterations: spread = {:.3f} {}".format(
                iterations, final_spread, unit))
        else:
            print("Points are coplanar (spread: {:.3f} {}).".format(initial_spread, unit))

        # --- Step 8: Rest the lowest contact point on the destination plane ---
        print("Seating objects onto destination plane...")
        level_onto_plane(copied_ids, copied_point_ids, target_plane)

        # --- Step 9: Surface mode - flip preview ---
        if mode == 'surface':
            print("\nPreview ready. Use 'Flip' (F) if the object should sit on the"
                  " other side of the surface; Enter to accept.")
            accepted = flip_preview(copied_ids, copied_point_ids, target_plane, cleanup)
            if not accepted:
                print("Cancelled by user.")
                return

        # --- Step 10: Report deviations ---
        deviation_report, max_deviation = report_point_deviations(
            copied_point_ids, target_plane)
        print("\n" + deviation_report)

        # --- Step 11: Clean up reference points ---
        print("\nCleaning up reference points...")
        cleanup.cleanup_points_only()

        # --- Done ---
        print("\n" + "=" * 50)
        print("SUCCESS!")
        print("=" * 50)
        if mode == 'xy':
            print("Oriented copy created {:.3f} {} away on +Y, resting on World XY.".format(
                offset_dist, unit))
        else:
            print("Oriented copy placed on the selected surface.")
        print("Original objects remain unchanged (all attributes preserved).")
        print("{} objects oriented with {} reference points.".format(
            len(copied_ids), original_point_count))

        if max_deviation > tol * 10:
            print("\nNOTE: Max point deviation from the destination plane is "
                  "{:.3f} {}.".format(max_deviation, unit))
            print("This may be due to non-coplanar reference points (curved surface).")

        rs.SelectObjects(copied_ids)


# =============================================================================
# SCRIPT RUNNER
# =============================================================================

if __name__ == "__main__":
    universal_orient()
