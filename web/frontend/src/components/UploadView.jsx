import { useCallback, useRef, useState } from "react";

export default function UploadView({ onUpload, loading, error }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    (e) => {
      const file = e.target.files[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          IDOC Resident Lookup
        </h2>
        <p className="text-gray-500">
          Upload an IDOC housing application PDF to look up resident information
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          w-full max-w-lg border-2 border-dashed rounded-2xl p-12
          flex flex-col items-center cursor-pointer transition-colors
          ${dragOver
            ? "border-amber-500 bg-amber-50"
            : "border-gray-300 bg-white hover:border-gray-400"
          }
          ${loading ? "opacity-50 pointer-events-none" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
        />

        {loading ? (
          <>
            <svg
              className="w-12 h-12 text-amber-500 animate-spin mb-4"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <p className="text-gray-700 font-medium">Processing document...</p>
            <p className="text-gray-400 text-sm mt-1">
              Extracting IDOC number and looking up resident
            </p>
          </>
        ) : (
          <>
            <svg
              className="w-12 h-12 text-gray-400 mb-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
              />
            </svg>
            <p className="text-gray-700 font-medium">
              Drop a PDF here or click to browse
            </p>
            <p className="text-gray-400 text-sm mt-1">
              IDOC housing applications (any format)
            </p>
          </>
        )}
      </div>

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-red-700 text-sm max-w-lg w-full">
          {error}
        </div>
      )}
    </div>
  );
}
