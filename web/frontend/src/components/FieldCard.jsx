export default function FieldCard({
  field,
  isSelected,
  onSelect,
  confidenceColor,
  confidenceLabel,
}) {
  const isCheckbox = field.kind === "checkbox_group";
  const displayValue = isCheckbox
    ? formatCheckboxValue(field.value)
    : field.value || "(empty)";

  const confPct = Math.round(field.confidence * 100);
  const colorClass = confidenceColor(field.confidence);
  const label = confidenceLabel(field.confidence);
  const review = field.review;

  function reviewBadgeClass() {
    if (!review?.needs_review) return "";
    if (review.severity === "high") return "text-red-700 bg-red-100 border-red-200";
    return "text-amber-700 bg-amber-100 border-amber-200";
  }

  return (
    <div
      onClick={onSelect}
      className={`
        rounded-lg border p-3 cursor-pointer transition-all
        ${isSelected
          ? "border-amber-400 bg-amber-50 ring-1 ring-amber-300"
          : "border-gray-200 bg-white hover:border-gray-300"
        }
      `}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-mono text-gray-400 truncate mb-1">
            {field.key}
          </p>
          <p className="text-sm text-gray-900 break-words">
            {displayValue}
          </p>
          {review?.needs_review && (
            <div className="mt-2">
              <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${reviewBadgeClass()}`}>
                {review.severity === "high" ? "Review required" : "Review recommended"}
              </span>
            </div>
          )}
        </div>

        <span
          className={`
            inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border shrink-0
            ${colorClass}
          `}
        >
          {confPct}%
          <span className="hidden sm:inline">{label}</span>
        </span>
      </div>

      {/* Expanded detail when selected */}
      {isSelected && (
        <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
          <DetailRow label="Source" value={field.source} />
          <DetailRow label="Kind" value={field.kind} />
          <DetailRow label="Confidence" value={`${(field.confidence * 100).toFixed(1)}%`} />
          {review?.needs_review && <DetailRow label="Review" value={review.status} />}
          {review?.message && (
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-md px-2 py-2">
              {review.message}
            </p>
          )}
          {Array.isArray(field.candidates) && field.candidates.length > 0 && (
            <div>
              <p className="text-xs text-gray-400 mb-1">Name Candidates</p>
              <div className="space-y-1">
                {field.candidates.slice(0, 5).map((candidate) => (
                  <div key={`${candidate.source}-${candidate.value}`} className="rounded-md bg-gray-50 border border-gray-200 px-2 py-2">
                    <div className="flex items-start justify-between gap-2 text-xs">
                      <span className="text-gray-500">{candidate.source}</span>
                      <span className="text-gray-500">{Math.round((candidate.confidence || 0) * 100)}%</span>
                    </div>
                    <div className="text-sm text-gray-800 break-words mt-1">{candidate.value || "(empty)"}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {isCheckbox && field.value?.scores && (
            <div>
              <p className="text-xs text-gray-400 mb-1">Checkbox Scores</p>
              <div className="space-y-1">
                {Object.entries(field.value.scores).map(([opt, score]) => (
                  <div key={opt} className="flex items-center gap-2">
                    <span className="text-xs text-gray-600 w-32 truncate">
                      {opt.replace(/_/g, " ")}
                    </span>
                    <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          score >= (field.value.threshold || 0.07)
                            ? "bg-amber-500"
                            : "bg-gray-300"
                        }`}
                        style={{ width: `${Math.min(score * 500, 100)}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500 w-12 text-right">
                      {(score * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                Threshold: {((field.value.threshold || 0.07) * 100).toFixed(0)}%
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-gray-400">{label}</span>
      <span className="text-gray-700 font-mono">{value}</span>
    </div>
  );
}

function formatCheckboxValue(value) {
  if (!value) return "(empty)";
  const selected = value.selected_options || [];
  if (selected.length === 0) return "(none selected)";
  return selected.map((s) => s.replace(/_/g, " ")).join(", ");
}
