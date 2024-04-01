interface Props {
  name: string;
  className?: string;
}

export function HeaderAtm({ name, className }: Props) {
  return (
    <h2 className={"text-lg font-bold" + (className ? ` ${className}` : "")}>
      {name}
    </h2>
  );
}
