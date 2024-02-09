import { ReactNode } from "react";

interface Props {
  label: string;
  disabled?: boolean;
  children: ReactNode;
}

export function LabeledConfig({ label, disabled = false, children }: Props) {
  return (
    <div
      className={
        "flex items-center justify-between text-xs" +
        (disabled ? " select-none opacity-50" : "")
      }
    >
      <label>{label}</label>
      {children}
    </div>
  );
}
