import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  dashed?: boolean;
  className?: string;
}

export function BorderBox({ children, dashed = false, className }: Props) {
  return (
    <div
      className={
        "rounded-lg border" +
        (dashed ? " border-dashed" : "") +
        (className ? ` ${className}` : "")
      }
    >
      {children}
    </div>
  );
}
