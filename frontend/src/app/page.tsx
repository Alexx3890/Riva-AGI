import Scene from './components/Scene';

export default function Home() {
  return (
    <main className="relative flex min-h-screen items-center justify-between p-8 overflow-hidden bg-[var(--color-dev-dark)]">
      {/* Background Spline Container */}
      <div className="absolute inset-0 z-0 flex items-center justify-center pointer-events-none">
        <div className="w-[800px] h-[800px] pointer-events-auto relative">
           <Scene />
        </div>
      </div>

      {/* Left Card */}
      <div className="neo-box z-10 w-80 text-sm leading-relaxed text-gray-300">
        <p>
          <strong className="text-white">Riva is our in-house machine learning model designed</strong> to explore real-world datasets, automate workflows, and power student research.
        </p>
      </div>

      {/* Center Label (Bottom) */}
      <div className="absolute bottom-12 left-1/2 -translate-x-1/2 z-10">
        <div className="neo-box-inset px-10 py-4 flex items-center justify-center">
          <h1 className="text-5xl font-black tracking-widest text-white uppercase drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]">
            Riva
          </h1>
        </div>
      </div>

      {/* Right Card */}
      <div className="neo-box z-10 w-72 text-sm text-right flex flex-col justify-center text-gray-300">
        <h3 className="font-bold tracking-widest text-white mb-2 text-xs">TRAIN. TEST. DEPLOY.</h3>
        <p className="text-xs text-gray-400">
          Optimized for <strong className="text-gray-300 font-medium">experimentation</strong> and rapid prototyping
        </p>
      </div>
    </main>
  );
}
