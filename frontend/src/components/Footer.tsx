import React from "react";

export function Footer() {
  return (
    <footer className="mt-auto border-t border-[var(--border)] bg-[var(--surface)] py-8 px-4 sm:px-6 lg:px-8 text-xs text-[var(--muted)] font-sans transition-colors">
      <div className="mx-auto max-w-7xl flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="space-y-1 text-center sm:text-left">
          <p className="font-serif text-[14px] text-[var(--foreground)]">
            GWLab &middot; Gravitational-Wave Parameter Estimation Laboratory
          </p>
          <p className="text-[12px] text-[var(--muted-light)]">
            Reference analysis environment &middot; LIGO Hanford (H1) design sensitivity &middot; Realization 01
          </p>
        </div>
        <div className="text-center sm:text-right text-[11px] text-[var(--muted-light)] space-y-0.5">
          <p>Scientific Engine: <code>gw-inference</code> (v1.0.0)</p>
          <p>Frontend: Next.js &middot; React &middot; Tailwind CSS</p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
