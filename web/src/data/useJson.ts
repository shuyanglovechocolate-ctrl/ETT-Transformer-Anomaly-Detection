import { useEffect, useState } from "react";

// Fetch a static JSON asset from public/data, respecting the Vite base path so
// it works both in dev (/) and on GitHub Pages (/<repo>/).
export function dataUrl(path: string): string {
  return `${import.meta.env.BASE_URL}data/${path}`;
}

type State<T> = { data: T | null; loading: boolean; error: string | null };

export function useJson<T>(path: string): State<T> {
  const [state, setState] = useState<State<T>>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let alive = true;
    setState({ data: null, loading: true, error: null });
    fetch(dataUrl(path))
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((data: T) => alive && setState({ data, loading: false, error: null }))
      .catch((e: Error) => alive && setState({ data: null, loading: false, error: e.message }));
    return () => {
      alive = false;
    };
  }, [path]);

  return state;
}
