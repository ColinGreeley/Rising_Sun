import { useRef, useState, useEffect } from "react";

function boxColor(confidence) {
  if (confidence >= 0.8) return "rgba(16, 185, 129, 0.35)";
  if (confidence >= 0.6) return "rgba(245, 158, 11, 0.35)";
  return "rgba(239, 68, 68, 0.35)";
}

function boxBorder(confidence) {
  if (confidence >= 0.8) return "rgba(16, 185, 129, 0.8)";
  if (confidence >= 0.6) return "rgba(245, 158, 11, 0.8)";
  return "rgba(239, 68, 68, 0.8)";
}

export default function PageOverlay({
  imageUrl,
  fields,
  showOverlays,
  selectedField,
  onSelectField,
}) {
  const containerRef = useRef(null);
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 });

  function handleImageLoad(e) {
    setImgSize({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
    });
  }

  // Build overlay rectangles from field specs
  const overlays = [];
  if (showOverlays) {
    for (const [key, field] of Object.entries(fields)) {
      if (field.kind === "checkbox_group" && field.value?.scores) {
        // For checkbox groups, we don't have individual box coords in the result,
        // but we still show one overlay per field using a rough region
        // (the actual boxes are in the template, not in field_results)
        continue;
      }
      // text / multiline_text fields have a box in the template, but it's not
      // in the extraction result directly. We'll use a placeholder approach:
      // show a label overlay on the right panel instead.
      overlays.push({
        key,
        confidence: field.confidence,
        isSelected: selectedField === key,
      });
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <img
        src={imageUrl}
        alt="Document page"
        onLoad={handleImageLoad}
        className="w-full h-auto block"
        crossOrigin="anonymous"
      />

      {/* Field highlight indicators along the left edge */}
      {showOverlays && imgSize.height > 0 && (
        <FieldIndicators
          fields={fields}
          selectedField={selectedField}
          onSelectField={onSelectField}
        />
      )}
    </div>
  );
}

function FieldIndicators({ fields, selectedField, onSelectField }) {
  // Group fields by approximate vertical position based on field order
  const fieldEntries = Object.entries(fields);
  if (fieldEntries.length === 0) return null;

  // Distribute field markers evenly along the page height
  const step = 100 / (fieldEntries.length + 1);

  return (
    <>
      {fieldEntries.map(([key, field], index) => {
        const top = step * (index + 1);
        const isSelected = selectedField === key;
        const conf = field.confidence;

        return (
          <div
            key={key}
            onClick={(e) => {
              e.stopPropagation();
              onSelectField(isSelected ? null : key);
            }}
            className="absolute cursor-pointer group"
            style={{ top: `${top}%`, right: 0 }}
          >
            <div
              className={`
                flex items-center gap-1 px-2 py-0.5 rounded-l-md text-xs font-mono
                transition-all whitespace-nowrap
                ${isSelected
                  ? "translate-x-0 opacity-100"
                  : "-translate-x-0 opacity-70 group-hover:opacity-100"
                }
              `}
              style={{
                backgroundColor: isSelected
                  ? boxColor(conf).replace("0.35", "0.9")
                  : boxColor(conf),
                borderLeft: `3px solid ${boxBorder(conf)}`,
              }}
            >
              <span className="text-gray-900 max-w-[120px] truncate">
                {key.split(".").pop().replace(/_/g, " ")}
              </span>
              <span className="text-gray-700 font-semibold">
                {Math.round(conf * 100)}%
              </span>
            </div>
          </div>
        );
      })}
    </>
  );
}
