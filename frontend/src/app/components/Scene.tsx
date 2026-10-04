'use client';

import Spline from '@splinetool/react-spline';

export default function Scene() {
  // Spline URL provided by user
  const splineUrl = "https://prod.spline.design/E1wDewq-L609UzAw/scene.splinecode";

  return (
    <div className="w-full h-full relative">
      <Spline scene={splineUrl} />
      {/* Mask to completely hide the "Built with Spline" watermark */}
      <div className="absolute bottom-0 right-0 w-[300px] h-[100px] bg-[#121212] z-50 pointer-events-none"></div>
    </div>
  );
}
