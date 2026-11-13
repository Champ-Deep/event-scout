import React, { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Marquee } from "@/components/ui/marquee";
import { q } from "../../lib/queries";
import { useConnection } from "./useConnection";
import "./events-admin.css";

// Installed 21st.dev lukacho/marquee, demo 1198. Host CSS owns sizing and motion.
export function EventTicker() {
  const online = useConnection();
  const events = useQuery(q.adminEvents());
  const planned = useQuery(q.plannerEvents());
  const names = [
    ...new Set([
      ...(events.data ?? []).map((e) => e.name),
      ...(planned.data ?? []).map((e) => e.name),
    ]),
  ];
  const [paused, setPaused] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const region = useRef<HTMLDivElement>(null);
  const fresh =
    online &&
    !events.isError &&
    !planned.isError &&
    !events.isFetching &&
    !planned.isFetching;
  useEffect(() => {
    const el = region.current;
    if (!el) return;
    const measure = () =>
      setOverflows(
        (el.querySelector(".es-event-group")?.getBoundingClientRect().width ??
          0) > el.clientWidth,
      );
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    const content = el.querySelector(".es-event-group");
    if (content) observer.observe(content);
    measure();
    return () => observer.disconnect();
  }, [names.join("\n")]);
  return (
    <section className="es-ticker" aria-label="Event names">
      <div
        ref={region}
        className="es-marquee min-w-0 flex-1"
        data-moving={overflows && fresh}
        data-paused={paused || !fresh}
      >
        {names.length ? (
          <Marquee aria-hidden="true" className="es-names" speed={70}>
            <span className="es-event-group">
              {names.map((name) => (
                <span key={name}>{name}</span>
              ))}
            </span>
          </Marquee>
        ) : (
          <span>
            {!online
              ? "Offline. No cached event names."
              : events.isPending || planned.isPending
                ? "Loading events..."
                : "No events yet. Add an event to start."}
          </span>
        )}
      </div>
      <button
        type="button"
        disabled={!fresh || !overflows}
        onClick={() => setPaused(!paused)}
      >
        {paused || !fresh ? "Resume ticker" : "Pause ticker"}
      </button>
      <details className="es-stationary">
        <summary>Event names</summary>
        <ul>
          {names.map((name) => (
            <li key={name}>{name}</li>
          ))}
        </ul>
      </details>
      {!online ? <span>Offline. Updates paused.</span> : null}
    </section>
  );
}
