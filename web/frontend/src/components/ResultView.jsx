export default function ResultView({ data }) {
  const info = data.idoc_info || {};
  const verification = data.verification || {};
  const hasError = !!info.error;

  return (
    <div className="space-y-6">
      {/* Verification banner */}
      <VerificationBanner verification={verification} />

      {/* PDF summary */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{data.filename}</h2>
            <p className="text-sm text-gray-500">
              {data.page_count} pages &middot; IDOC #{data.idoc_number || "not found"}
              {data.extraction_method && (
                <span className="ml-2 text-xs text-gray-400">({data.extraction_method})</span>
              )}
            </p>
          </div>
          {info.idoc_url && (
            <a
              href={info.idoc_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-amber-600 hover:text-amber-800 underline"
            >
              View on IDOC website &rarr;
            </a>
          )}
        </div>
      </div>

      {hasError ? (
        <div className="rounded-xl border border-red-200 bg-red-50 p-5">
          <p className="text-red-800 font-medium">Lookup failed</p>
          <p className="text-sm text-red-600 mt-1">{info.error}</p>
        </div>
      ) : (
        <>
          {/* Resident info */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-4">
              Resident Information
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <InfoField label="Name" value={info.name} />
              <InfoField label="IDOC Number" value={info.idoc_number} />
              <InfoField label="Status" value={info.status} />
              <InfoField label="Age" value={info.age} />
              <InfoField label="Phone" value={info.phone} />
            </div>
            {info.mailing_address && (
              <div className="mt-4">
                <InfoField label="Mailing Address" value={info.mailing_address} />
              </div>
            )}
          </div>

          {/* Sentences */}
          {info.sentences && info.sentences.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-4">
                Sentence Information
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                      <th className="py-2 pr-4">Offense</th>
                      <th className="py-2 pr-4">County</th>
                      <th className="py-2 pr-4">Case #</th>
                      <th className="py-2 pr-4">Status</th>
                      <th className="py-2 pr-4">Released</th>
                      <th className="py-2">Termination</th>
                    </tr>
                  </thead>
                  <tbody>
                    {info.sentences.map((s, i) => (
                      <tr key={i} className="border-b border-gray-100">
                        <td className="py-2 pr-4 text-gray-900">{s.offense}</td>
                        <td className="py-2 pr-4 text-gray-700">{s.county}</td>
                        <td className="py-2 pr-4 text-gray-700 font-mono text-xs">{s.case_number}</td>
                        <td className="py-2 pr-4">
                          <StatusBadge status={s.sentence_status} />
                        </td>
                        <td className="py-2 pr-4 text-gray-700">{s.released_to_supervision || "N/A"}</td>
                        <td className="py-2 text-gray-700">{s.termination_date || "N/A"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Diagnostic details (collapsible) */}
      <details className="bg-white rounded-xl border border-gray-200 p-5">
        <summary className="text-sm font-semibold uppercase tracking-wider text-gray-500 cursor-pointer select-none">
          Diagnostic Details
        </summary>
        <div className="mt-3 space-y-2 text-sm text-gray-700">
          {verification.extraction_method && (
            <p>
              <span className="font-medium">Extraction method:</span>{" "}
              <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono">{verification.extraction_method}</code>
            </p>
          )}
          {verification.raw_capture && (
            <p>
              <span className="font-medium">Raw OCR capture:</span>{" "}
              <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono">{verification.raw_capture}</code>
            </p>
          )}
          {verification.candidates_tried && verification.candidates_tried.length > 0 && (
            <>
              <p>
                <span className="font-medium">Candidates tried:</span>
              </p>
              <ul className="ml-4 space-y-1">
                {verification.candidates_tried.map((c, i) => (
                  <li key={i} className="font-mono text-xs">
                    {c.number}{" "}
                    <span className={c.found ? "text-green-600" : "text-red-500"}>
                      {c.found ? "found in database" : "not found"}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {verification.name_crosscheck && (verification.name_crosscheck.ocr_name || verification.name_crosscheck.filename_name) && (
            <div className="mt-2 pt-2 border-t border-gray-100">
              <p className="font-medium">Name cross-check:</p>
              {verification.name_crosscheck.filename_name && (
                <p className="mt-1">
                  <span className="text-gray-500">Filename name:</span>{" "}
                  <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono">{verification.name_crosscheck.filename_name}</code>
                </p>
              )}
              {verification.name_crosscheck.ocr_name && (
                <p className="mt-0.5">
                  <span className="text-gray-500">Form name (OCR):</span>{" "}
                  <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono">{verification.name_crosscheck.ocr_name}</code>
                </p>
              )}
              {verification.name_crosscheck.idoc_name && (
                <p className="mt-0.5">
                  <span className="text-gray-500">IDOC website name:</span>{" "}
                  <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono">{verification.name_crosscheck.idoc_name}</code>
                  {" "}
                  <span className={verification.name_crosscheck.match ? "text-green-600" : "text-amber-600"}>
                    {verification.name_crosscheck.match
                      ? `✓ ${verification.name_crosscheck.match_level || "match"}`
                      : "⚠ mismatch"}
                  </span>
                </p>
              )}
            </div>
          )}
        </div>
      </details>
    </div>
  );
}

function InfoField({ label, value }) {
  return (
    <div>
      <dt className="text-xs font-medium text-gray-500">{label}</dt>
      <dd className="mt-0.5 text-sm text-gray-900">{value || "N/A"}</dd>
    </div>
  );
}

function StatusBadge({ status }) {
  const color = status === "Active"
    ? "bg-green-100 text-green-800"
    : "bg-gray-100 text-gray-800";
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>
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
      label: "Verification Failed",
      dot: "bg-red-500",
    },
  };

  const c = colors[verification.status] || colors.red;

  return (
    <div className={`rounded-xl border ${c.border} ${c.bg} p-4`}>
      <div className="flex items-start gap-3">
        <span className={`mt-0.5 h-3 w-3 rounded-full ${c.dot} flex-shrink-0`}></span>
        <div>
          <p className={`font-semibold ${c.text}`}>{c.label}</p>
          <p className={`text-sm mt-0.5 ${c.text} opacity-80`}>{verification.reason}</p>
        </div>
      </div>
    </div>
  );
}
