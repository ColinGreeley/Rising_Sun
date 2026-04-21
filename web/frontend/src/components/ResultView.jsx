import { useState } from "react";

export default function ResultView({ data }) {
  const info = data.idoc_info || {};
  const verification = data.verification || {};
  const rso = data.rso || null;
  const cropImages = data.crop_images || null;
  const candidateResults = data.candidate_results || [];
  const hasError = !!info.error;
  
  const [zoomedImage, setZoomedImage] = useState(null);

  return (
    <div className="space-y-6 relative">
      {/* Zoom Lightbox */}
      {zoomedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={() => setZoomedImage(null)}
        >
           <button 
             className="absolute top-4 right-4 text-white hover:text-gray-300"
             onClick={() => setZoomedImage(null)}
           >
             <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
           </button>
           <img 
             src={zoomedImage.src} 
             alt={zoomedImage.alt} 
             className="max-h-[90vh] max-w-[90vw] object-contain rounded bg-white p-2 shadow-2xl" 
             onClick={(e) => e.stopPropagation()} 
           />
        </div>
      )}

      {/* Verification banner */}
      <VerificationBanner verification={verification} />

      {/* PDF summary */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{data.filename}</h2>
            <p className="text-sm text-gray-500">
              {data.page_count} pages &middot; IDOC #{data.idoc_number || "not found"}
              {data.extraction_method && (
                <span className="ml-2 text-xs text-gray-400 font-mono bg-gray-100 px-1 rounded">
                  {data.extraction_method}
                </span>
              )}
            </p>
          </div>
          {info.idoc_url && (
            <a
              href={info.idoc_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-amber-600 hover:text-amber-800 underline font-medium"
            >
              View on IDOC website &rarr;
            </a>
          )}
        </div>
      </div>

      {hasError ? (
        <div className="rounded-xl border border-red-200 bg-red-50 p-5 shadow-sm">
          <p className="text-red-800 font-bold flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
            Lookup failed
          </p>
          <p className="text-sm text-red-600 mt-2 font-mono">{info.error}</p>
        </div>
      ) : (
        <>
          {/* Resident info */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-4 border-b pb-2">
              Resident Information
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <InfoField label="Name" initialValue={info.name} />
              <InfoField label="IDOC Number" initialValue={info.idoc_number} />
              <InfoField label="Status" initialValue={info.status} />
              <InfoField label="Age" initialValue={info.age} />
              <InfoField label="Phone" initialValue={info.phone} />
            </div>
            {info.mailing_address && (
              <div className="mt-6">
                <InfoField label="Mailing Address" initialValue={info.mailing_address} />
              </div>
            )}
            
            {rso && (
              <div className="mt-8 pt-5 border-t border-gray-100">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500">Sex Offender Registry</h3>
                </div>
                <div className="flex items-center gap-4 flex-wrap">
                  <span
                    className={`inline-flex items-center rounded-lg px-5 py-2.5 text-lg font-bold shadow-sm ${
                      rso.is_rso
                        ? "bg-red-100 text-red-800 border border-red-200"
                        : rso.needs_review
                          ? "bg-amber-100 text-amber-800 border border-amber-200"
                          : "bg-green-100 text-green-800 border border-green-200"
                    }`}
                  >
                    {rso.is_rso ? "⚠ RSO: Yes" : rso.needs_review ? "⁈ RSO: Needs Review" : "✓ RSO: No"}
                  </span>
                  {rso.confidence != null && (
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-700">
                        {Math.round(rso.confidence * 100)}% Confidence
                      </span>
                      {rso.method && (
                        <span className="text-xs text-gray-400 font-mono">Model: {rso.method.replace("template_", "v")}</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Sentences */}
          {info.sentences && info.sentences.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-4 border-b pb-2">
                Sentence Information
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                      <th className="py-3 pr-4">Offense</th>
                      <th className="py-3 pr-4">County</th>
                      <th className="py-3 pr-4">Case #</th>
                      <th className="py-3 pr-4">Status</th>
                      <th className="py-3 pr-4">Released</th>
                      <th className="py-3">Termination</th>
                    </tr>
                  </thead>
                  <tbody>
                    {info.sentences.map((s, i) => (
                      <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                        <td className="py-3 pr-4 text-gray-900 font-medium">{s.offense}</td>
                        <td className="py-3 pr-4 text-gray-700">{s.county}</td>
                        <td className="py-3 pr-4 text-gray-700 font-mono text-xs">{s.case_number}</td>
                        <td className="py-3 pr-4">
                          <StatusBadge status={s.sentence_status} />
                        </td>
                        <td className="py-3 pr-4 text-gray-700">{s.released_to_supervision || "N/A"}</td>
                        <td className="py-3 text-gray-700">{s.termination_date || "N/A"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {!hasError && candidateResults.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4 border-b pb-2">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500">
              Candidate Ranking
            </h3>
            <p className="text-xs text-gray-500">
              Final selection is the successful IDOC database hit whose name most closely matches the OCR name.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                  <th className="py-3 pr-4">Rank</th>
                  <th className="py-3 pr-4">IDOC #</th>
                  <th className="py-3 pr-4">Database Name</th>
                  <th className="py-3 pr-4">Matched OCR Name</th>
                  <th className="py-3 pr-4">Match</th>
                  <th className="py-3 pr-4">Score</th>
                  <th className="py-3">Picked</th>
                </tr>
              </thead>
              <tbody>
                {candidateResults.map((candidate, index) => {
                  const isSelected = candidate.idoc_number === data.idoc_number;
                  const matchStyles = {
                    exact: "bg-green-100 text-green-800 border-green-200",
                    strong: "bg-emerald-100 text-emerald-800 border-emerald-200",
                    partial: "bg-amber-100 text-amber-800 border-amber-200",
                    weak: "bg-orange-100 text-orange-800 border-orange-200",
                    none: "bg-red-100 text-red-800 border-red-200",
                  };
                  const matchClass = matchStyles[candidate.match_level] || matchStyles.none;
                  return (
                    <tr key={`${candidate.idoc_number}-${index}`} className={`border-b border-gray-50 ${isSelected ? "bg-amber-50/60" : "hover:bg-gray-50"}`}>
                      <td className="py-3 pr-4 text-gray-500 font-medium">#{index + 1}</td>
                      <td className="py-3 pr-4 font-mono text-xs text-gray-900">{candidate.idoc_number}</td>
                      <td className="py-3 pr-4 text-gray-900 font-medium">{candidate.idoc_name || "Unknown"}</td>
                      <td className="py-3 pr-4 text-gray-700 font-mono text-xs">{candidate.matched_ocr_name || "N/A"}</td>
                      <td className="py-3 pr-4">
                        <span className={`inline-flex items-center border rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${matchClass}`}>
                          {candidate.match_level}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-gray-700 font-mono text-xs">{candidate.match_score?.toFixed ? candidate.match_score.toFixed(1) : candidate.match_score}</td>
                      <td className="py-3">
                        {isSelected ? (
                          <span className="inline-flex items-center border border-amber-200 rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-wide bg-amber-100 text-amber-800">
                            Selected
                          </span>
                        ) : (
                          <span className="text-xs text-gray-400">Alternate</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Visual Evidence (Crops) */}
      {cropImages && (cropImages.idoc_field || cropImages.name_field || cropImages.rso_field) && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-4 border-b pb-2">
            Extraction Visual Evidence
          </h3>
          <div className="flex flex-col gap-6">
            {cropImages.name_field && (
              <div className="group">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-700">Applicant Name Region</p>
                  <p className="text-xs text-gray-400 hidden group-hover:block transition-all">Click to zoom</p>
                </div>
                <div 
                  className="border-2 border-gray-100 rounded-lg overflow-hidden cursor-zoom-in hover:border-amber-300 transition-colors"
                  onClick={() => setZoomedImage({ src: cropImages.name_field, alt: "Applicant Name Region" })}
                >
                  <img
                    src={cropImages.name_field}
                    alt="Applicant name field crop"
                    className="w-full object-contain bg-gray-50 p-2 max-h-[150px] mix-blend-multiply"
                  />
                </div>
              </div>
            )}

            {cropImages.idoc_field && (
              <div className="group">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-700">IDOC Number Region</p>
                  <p className="text-xs text-gray-400 hidden group-hover:block transition-all">Click to zoom</p>
                </div>
                <div 
                  className="border-2 border-gray-100 rounded-lg overflow-hidden cursor-zoom-in hover:border-amber-300 transition-colors"
                  onClick={() => setZoomedImage({ src: cropImages.idoc_field, alt: "IDOC Number Region" })}
                >
                  <img
                    src={cropImages.idoc_field}
                    alt="IDOC number field crop"
                    className="w-full object-contain bg-gray-50 p-2 max-h-[100px] mix-blend-multiply"
                  />
                </div>
              </div>
            )}

            {cropImages.rso_field && (
              <div className="group">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-700">Sex Offender Registration Region</p>
                  <p className="text-xs text-gray-400 hidden group-hover:block transition-all">Click to zoom</p>
                </div>
                <div 
                  className="border-2 border-gray-100 rounded-lg overflow-hidden cursor-zoom-in hover:border-amber-300 transition-colors"
                  onClick={() => setZoomedImage({ src: cropImages.rso_field, alt: "RSO checkbox crop" })}
                >
                  <img
                    src={cropImages.rso_field}
                    alt="RSO checkbox crop"
                    className="w-full object-contain bg-gray-50 p-2 max-h-[150px] mix-blend-multiply"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Diagnostic details (collapsible) */}
      <details className="bg-white rounded-xl border border-gray-200 p-5 group shadow-sm">
        <summary className="text-sm font-semibold uppercase tracking-wider text-gray-500 cursor-pointer select-none flex items-center justify-between">
          <span>Diagnostic Details</span>
          <svg className="w-4 h-4 text-gray-400 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
        </summary>
        <div className="mt-4 space-y-3 text-sm text-gray-700">
          {verification.extraction_method && (
            <p className="flex justify-between items-center py-1 border-b border-gray-50">
              <span className="font-medium">Extraction method:</span>{" "}
              <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-amber-800">{verification.extraction_method}</code>
            </p>
          )}
          {verification.raw_capture && (
            <p className="flex justify-between items-center py-1 border-b border-gray-50">
              <span className="font-medium">Raw OCR capture:</span>{" "}
              <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-amber-800">{verification.raw_capture}</code>
            </p>
          )}
          {verification.candidates_tried && verification.candidates_tried.length > 0 && (
            <div className="py-2">
              <p className="font-medium mb-2">Candidates tried:</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {verification.candidates_tried.map((c, i) => (
                  <div key={i} className="flex items-center gap-2 bg-gray-50 p-2 rounded border border-gray-100">
                    <span className="font-mono text-xs text-amber-900 border border-gray-200 rounded bg-white px-1">{c.number}</span>
                    <span className={c.found ? "text-green-600 text-xs font-medium" : "text-gray-500 text-xs"}>
                      {c.found ? "✓ found" : "✗ missed"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {verification.name_crosscheck && verification.name_crosscheck.ocr_name && (
            <div className="mt-3 pt-3 border-t border-gray-100">
              <p className="font-medium mb-2">Name Cross-check Results:</p>
              <div className="bg-gray-50 rounded p-3 space-y-2 border border-gray-100">
                <p className="flex justify-between text-xs">
                  <span className="text-gray-500">Source PDF OCR</span>
                  <span className="font-mono font-medium text-gray-900">{verification.name_crosscheck.ocr_name}</span>
                </p>
                <p className="flex justify-between text-xs">
                  <span className="text-gray-500">IDOC Database</span>
                  <span className="font-mono font-medium text-gray-900">{verification.name_crosscheck.idoc_name}</span>
                </p>
                {verification.name_crosscheck.ocr_names && verification.name_crosscheck.ocr_names.length > 1 && (
                  <p className="flex justify-between gap-4 text-xs">
                    <span className="text-gray-500">OCR Candidate Names</span>
                    <span className="font-mono font-medium text-gray-700 text-right">
                      {verification.name_crosscheck.ocr_names.join(" | ")}
                    </span>
                  </p>
                )}
                {verification.name_crosscheck.selected_match_score != null && (
                  <p className="flex justify-between text-xs">
                    <span className="text-gray-500">Selected Match Score</span>
                    <span className="font-mono font-medium text-gray-900">{verification.name_crosscheck.selected_match_score}</span>
                  </p>
                )}
                <div className="pt-2 mt-2 border-t border-gray-200 flex justify-end">
                   <span className={`px-2 py-0.5 rounded text-xs font-bold ${verification.name_crosscheck.match ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                    {verification.name_crosscheck.match
                      ? `MATCH (${verification.name_crosscheck.match_level || "EXACT"})`
                      : "⚠ MISMATCH"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}

// Editable InfoField
function InfoField({ label, initialValue }) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(initialValue || "");

  const handleSave = () => setIsEditing(false);
  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") {
      setValue(initialValue || "");
      setIsEditing(false);
    }
  };

  return (
    <div className="group flex flex-col">
      <dt className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">{label}</dt>
      <dd className="relative flex items-center min-h-[32px]">
        {isEditing ? (
          <div className="flex w-full items-center gap-2">
            <input 
              autoFocus
              type="text" 
              value={value} 
              onChange={e => setValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={handleSave}
              className="w-full text-sm font-medium text-gray-900 border-2 border-amber-300 rounded px-2 py-1 outline-none focus:ring-2 focus:ring-amber-500/20"
            />
          </div>
        ) : (
          <div 
            className="flex items-center gap-2 w-full cursor-text hover:bg-gray-50 p-1 -ml-1 rounded transition-colors"
            onClick={() => setIsEditing(true)}
            title="Click to edit"
          >
            <span className={`text-base font-medium ${value ? "text-gray-900" : "text-gray-400 italic"}`}>
              {value || "Not Found"}
            </span>
            <button 
              className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-amber-500 transition-opacity ml-auto"
              aria-label="Edit field"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
            </button>
          </div>
        )}
      </dd>
    </div>
  );
}

function StatusBadge({ status }) {
  const color = status === "Active"
    ? "bg-green-100 text-green-800 border-green-200"
    : "bg-gray-100 text-gray-800 border-gray-200";
  return (
    <span className={`inline-flex items-center border rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${color}`}>
      {status || "N/A"}
    </span>
  );
}

function VerificationBanner({ verification }) {
  if (!verification || !verification.status) return null;

  const colors = {
    green: {
      border: "border-green-200",
      bg: "bg-green-50",
      icon: "text-green-600",
      text: "text-green-800",
      label: "Verified",
      dot: "bg-green-500",
    },
    yellow: {
      border: "border-amber-200",
      bg: "bg-amber-50",
      icon: "text-amber-600",
      text: "text-amber-800",
      label: "Needs Review",
      dot: "bg-amber-500",
    },
    red: {
      border: "border-red-200",
      bg: "bg-red-50",
      icon: "text-red-600",
      text: "text-red-800",
      label: "Unverified",
      dot: "bg-red-500",
    },
  };

  const style = colors[verification.status] || colors.yellow;

  return (
    <div className={`rounded-xl border ${style.border} ${style.bg} p-4 shadow-sm`}>
      <div className="flex items-center gap-3">
        <span className="relative flex h-3 w-3">
          {(verification.status === 'yellow' || verification.status === 'red') && (
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${style.dot}`}></span>
          )}
          <span className={`relative inline-flex rounded-full h-3 w-3 ${style.dot}`}></span>
        </span>
        <h3 className={`text-sm font-bold uppercase tracking-wider ${style.text}`}>{style.label}</h3>
      </div>
      {(verification.reason || verification.message) && (
        <p className={`mt-2 text-sm ml-6 ${style.text} opacity-90 font-medium`}>{verification.reason || verification.message}</p>
      )}
    </div>
  );
}

export function SkeletonResultView() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Banner Skeleton */}
      <div className="h-16 bg-gray-200 rounded-xl w-full"></div>
      
      {/* Title Skeleton */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
        <div className="h-6 bg-gray-200 rounded w-1/2 mb-3"></div>
        <div className="h-4 bg-gray-200 rounded w-1/3"></div>
      </div>
      
      {/* Body Skeleton */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-6">
        <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="space-y-2"><div className="h-3 bg-gray-200 w-1/3 rounded"></div><div className="h-5 bg-gray-200 w-3/4 rounded"></div></div>
          <div className="space-y-2"><div className="h-3 bg-gray-200 w-1/3 rounded"></div><div className="h-5 bg-gray-200 w-1/2 rounded"></div></div>
          <div className="space-y-2"><div className="h-3 bg-gray-200 w-1/3 rounded"></div><div className="h-5 bg-gray-200 w-2/3 rounded"></div></div>
          <div className="space-y-2"><div className="h-3 bg-gray-200 w-1/3 rounded"></div><div className="h-5 bg-gray-200 w-1/4 rounded"></div></div>
        </div>
        <div className="pt-6 border-t border-gray-100 flex items-center justify-between">
          <div className="h-12 bg-gray-200 rounded-lg w-1/3"></div>
        </div>
      </div>

      {/* Image Skeletons */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
        <div className="h-4 bg-gray-200 rounded w-1/4 mb-6"></div>
        <div className="space-y-4">
           <div className="h-32 bg-gray-200 rounded-lg w-full"></div>
           <div className="h-32 bg-gray-200 rounded-lg w-full"></div>
        </div>
      </div>
    </div>
  );
}
